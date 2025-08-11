import json
import logging
import os
import time
import uuid
import re
import urllib.parse
from typing import Dict, List, AsyncGenerator, Tuple
import asyncio

import httpx
import pybase64
from google import genai
from google.genai import types
from dotenv import load_dotenv

import modal

# Load environment variables from .env file
load_dotenv("/.env")

# Import the Master Realtor prompt template
import sys

sys.path.append("/")
from realtor_prompt import MASTER_REALTOR_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("property-image-processor")

# Define the Modal image with all required dependencies
image = modal.Image.debian_slim().pip_install(
    [
        "boto3",
        "httpx",
        "pybase64",
        "google-genai",  # Updated package name
        "python-dotenv",
        "fastapi[standard]",
    ]
)

# Mount the .env file and realtor_prompt.py
mounts = [
    modal.Mount.from_local_file(".env", remote_path="/.env"),
    modal.Mount.from_local_file("realtor_prompt.py", remote_path="/realtor_prompt.py"),
]


class PropertyImageProcessor:
    def __init__(
        self,
        gemini_api_key: str = None,
        gemini_model: str = None,
        rate_limit: int = 200,
        max_retries: int = 3,
        max_size_mb: float = 50.0,
        max_concurrent: int = 100,
    ):
        # Load from environment variables or use provided values
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.environ.get(
            "GEMINI_MODEL", "gemini-2.5-pro"
        )

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY must be provided either as parameter or in environment variable"
            )

        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.max_size_mb = max_size_mb
        self.max_concurrent = max_concurrent

        # Configure Gemini client
        self.client = genai.Client(api_key=self.gemini_api_key)

        # Rate limiting (200 per minute = ~3.33 per second) with thread safety
        self.last_request_times = []
        self._rate_limit_lock = asyncio.Lock()

        # Setup logging for MLS tracking
        self.logger = logger

    def extract_properties_from_json(self, data: Dict) -> Dict[str, Dict]:
        """Extract property URLs, their JPEG images, and property details from JSON data."""
        self.logger.info("Starting property extraction from JSON data")
        properties_dict = {}

        # Check if data has 'properties' key
        if "properties" in data:
            properties = data["properties"]
        else:
            properties = data

        for property_url, property_data in properties.items():
            if isinstance(property_data, dict) and "images" in property_data:
                # Filter only JPEG images
                jpeg_images = [
                    img_url
                    for img_url in property_data["images"]
                    if img_url.lower().endswith((".jpeg", ".jpg"))
                ]

                if jpeg_images:
                    # Extract property details
                    property_details = self._extract_property_details(property_data)

                    # Add MLS info directly to the property data for visibility
                    mls_info = property_details.get("mls_number", "NONE")
                    is_genuine = property_details.get("mls_is_genuine", False)

                    properties_dict[property_url] = {
                        "mls_number": mls_info,
                        "mls_is_genuine": is_genuine,
                        "mls_status": "genuine" if is_genuine else "generated",
                        "images": jpeg_images,
                        "details": property_details,
                    }

                    # Log MLS extraction result
                    self.logger.info(
                        f"Property: {property_url[:50]}... | MLS: {mls_info} ({'genuine' if is_genuine else 'generated'})"
                    )

        self.logger.info(f"Extracted {len(properties_dict)} properties with images")
        return properties_dict

    def extract_removed_properties_mls(self, data: Dict) -> List[str]:
        """Extract MLS numbers from removed properties."""
        self.logger.info("Starting MLS extraction from removed properties")
        mls_numbers = []

        if "removed_properties" not in data:
            self.logger.info("No removed_properties section found")
            return mls_numbers

        removed_properties = data["removed_properties"]

        for property_url, property_data in removed_properties.items():
            if isinstance(property_data, dict) and "title" in property_data:
                title = property_data["title"]
                mls_number, is_genuine = self._extract_mls_number(title)

                # Only add genuine MLS numbers for removed properties
                if is_genuine:
                    mls_numbers.append(mls_number)
                    self.logger.info(
                        f"Removed property MLS: {mls_number} from {property_url[:50]}..."
                    )
                else:
                    self.logger.info(
                        f"Removed property (no valid MLS): {property_url[:50]}..."
                    )

        self.logger.info(
            f"Extracted {len(mls_numbers)} MLS numbers from removed properties"
        )
        return mls_numbers

    def _extract_mls_number(self, title: str) -> Tuple[str, bool]:
        """Extract MLS number from title. Returns (mls_number, is_genuine)."""
        self.logger.debug(f"MLS extraction from title: '{title}'")

        if not title or not isinstance(title, str):
            generated_mls = f"GEN{uuid.uuid4().hex[:8].upper()}"
            self.logger.debug(f"Empty/invalid title, generated: {generated_mls}")
            return generated_mls, False

        # Split by pipe and get the last part (most common MLS pattern)
        parts = title.split("|")
        self.logger.debug(f"Title split into {len(parts)} parts: {parts}")

        if len(parts) >= 2:
            potential_mls = parts[-1].strip()
        else:
            # Try other common separators
            for separator in ["-", ":", "#", "•"]:
                if separator in title:
                    parts = title.split(separator)
                    potential_mls = parts[-1].strip()
                    break
            else:
                # No separator found, check if entire title could be MLS
                potential_mls = title.strip()

        self.logger.debug(f"Potential MLS after extraction: '{potential_mls}'")

        # Clean up potential MLS - remove common prefixes/suffixes
        potential_mls = re.sub(
            r"^(MLS|#|ID|REF)[:.\s]*", "", potential_mls, flags=re.IGNORECASE
        )
        potential_mls = re.sub(
            r"[^\w]", "", potential_mls
        )  # Remove all non-alphanumeric

        self.logger.debug(f"Cleaned potential MLS: '{potential_mls}'")

        # Validate if it looks like a genuine MLS number
        if self._is_valid_mls(potential_mls):
            self.logger.info(f"✅ Found genuine MLS: {potential_mls}")
            return potential_mls, True
        else:
            generated_mls = f"GEN{uuid.uuid4().hex[:8].upper()}"
            self.logger.info(f"❌ No valid MLS found, generated: {generated_mls}")
            return generated_mls, False

    def _is_valid_mls(self, mls_candidate: str) -> bool:
        """Validate if a string looks like a genuine MLS number."""
        if not mls_candidate or len(mls_candidate) < 4:
            return False

        # Check basic alphanumeric requirement
        if not re.match(r"^[A-Za-z0-9]+$", mls_candidate):
            return False

        # Must have both letters and numbers (typical MLS pattern)
        has_letters = bool(re.search(r"[A-Za-z]", mls_candidate))
        has_numbers = bool(re.search(r"[0-9]", mls_candidate))

        # Length should be reasonable (typically 4-15 characters)
        reasonable_length = 4 <= len(mls_candidate) <= 15

        # Additional patterns that might indicate a genuine MLS
        common_patterns = [
            r"^[A-Z]{1,3}\d{4,}",  # Letters followed by numbers (e.g., AB123456)
            r"^\d{4,}[A-Z]{1,3}",  # Numbers followed by letters (e.g., 123456AB)
            r"^[A-Z0-9]{6,}$",  # Mixed alphanumeric, reasonable length
        ]

        pattern_match = any(
            re.match(pattern, mls_candidate.upper()) for pattern in common_patterns
        )

        is_valid = has_letters and has_numbers and reasonable_length and pattern_match

        self.logger.debug(
            f"MLS validation for '{mls_candidate}': letters={has_letters}, numbers={has_numbers}, length={reasonable_length}, pattern={pattern_match} -> {is_valid}"
        )

        return is_valid

    def _extract_property_details(self, property_data: Dict) -> Dict:
        """Extract key property details from JSON data."""
        details = {}

        # Property address
        if "property_address" in property_data:
            details["address"] = property_data["property_address"]
        elif "address" in property_data and isinstance(property_data["address"], dict):
            details["address"] = property_data["address"].get("streetAddress", "")

        # Listed price
        if "price" in property_data:
            details["price"] = property_data["price"]
        if "currency" in property_data:
            details["currency"] = property_data["currency"]

        # MLS Description
        if "description" in property_data:
            details["description"] = property_data["description"]
        elif "meta_description" in property_data:
            details["description"] = property_data["meta_description"]

        # Structured details
        if "bedrooms" in property_data:
            details["bedrooms"] = property_data["bedrooms"]
        if "bathrooms" in property_data:
            details["bathrooms"] = property_data["bathrooms"]
        if "property_type" in property_data:
            details["property_type"] = property_data["property_type"]

        # Extract MLS number from title - IMPROVED LOGIC
        title = property_data.get("title", "")
        mls_number, is_genuine = self._extract_mls_number(title)
        details["mls_number"] = mls_number
        details["mls_is_genuine"] = is_genuine

        return details

    def _create_prompt_with_details(self, property_details: Dict) -> str:
        """Create comprehensive Master Realtor Assistant prompt with property details."""
        address = property_details.get("address", "Not specified")

        # Format price
        price_info = "Not specified"
        if property_details.get("price"):
            currency = property_details.get("currency", "")
            price_formatted = (
                f"${property_details['price']:,}"
                if isinstance(property_details["price"], (int, float))
                else property_details["price"]
            )
            price_info = f"{price_formatted} {currency}".strip()

        description = property_details.get("description", "Not provided")

        # Build structured details
        structured_details = []
        if property_details.get("bedrooms"):
            structured_details.append(f"Bedrooms: {property_details['bedrooms']}")
        if property_details.get("bathrooms"):
            structured_details.append(f"Bathrooms: {property_details['bathrooms']}")
        if property_details.get("property_type"):
            structured_details.append(f"Type: {property_details['property_type']}")

        # IMPROVED: Always include MLS number with context about whether it's genuine
        mls_number = property_details.get("mls_number", "Unknown")
        is_genuine = property_details.get("mls_is_genuine", False)
        if is_genuine:
            structured_details.append(f"MLS: {mls_number}")
        else:
            structured_details.append(f"Property ID: {mls_number} (system-generated)")

        structured_details_text = (
            " | ".join(structured_details) if structured_details else "Not specified"
        )

        # Fill in the placeholders in the Master Realtor prompt
        full_prompt = MASTER_REALTOR_PROMPT.format(
            property_address=address,
            property_price=price_info,
            property_description=description,
            property_details=structured_details_text,
        )

        return full_prompt

    async def process_images(
        self, client: httpx.AsyncClient, image_urls: List[str]
    ) -> AsyncGenerator[Tuple[str, int], None]:
        """Generator that downloads and encodes images one by one."""
        for image_url in image_urls:
            try:
                # Download image with retries
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.get(image_url, timeout=30.0)
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if attempt < self.max_retries:
                            wait_time = 2**attempt
                            await asyncio.sleep(wait_time)
                        else:
                            raise e

                # Encode to base64
                image_base64 = pybase64.b64encode(response.content).decode("utf-8")
                image_size_bytes = len(image_base64.encode("utf-8"))

                yield image_base64, image_size_bytes

            except Exception as e:
                self.logger.error(f"Failed to process image {image_url}: {e}")
                continue

    async def process_single_property(
        self,
        client: httpx.AsyncClient,
        property_url: str,
        property_data: Dict,
    ) -> Dict:
        """Process a single property with memory-efficient image handling."""
        try:
            images_batch = []
            batch_size_bytes = 0
            max_size_bytes = self.max_size_mb * 1024 * 1024
            images_processed_count = 0

            # Generator: Process images one by one
            async for encoded_image, size_bytes in self.process_images(
                client, property_data["images"]
            ):
                # Check if adding this image would exceed size limit
                if batch_size_bytes + size_bytes > max_size_bytes and images_batch:
                    break

                images_batch.append(encoded_image)
                batch_size_bytes += size_bytes
                images_processed_count += 1

            if not images_batch:
                error_msg = "No images could be processed"
                return self._create_error_result(property_url, property_data, error_msg)

            # Auto-trigger API call when batch ready
            description = await self._call_gemini_api(
                images_batch, property_data["details"]
            )

            result = {
                "property_url": property_url,
                "property_details": self._format_property_details(
                    property_data["details"]
                ),
                "processing_info": {
                    "images_processed": images_processed_count,
                    "images_analyzed": property_data["images"][:images_processed_count],
                    "status": "success",
                    "error_message": None,
                },
                "ai_analysis_raw": description,
            }

            # Clear memory immediately
            del images_batch
            return result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to process {property_url}: {error_msg}")
            return self._create_error_result(property_url, property_data, error_msg)

    def _create_error_result(
        self, property_url: str, property_data: Dict, error_msg: str
    ) -> Dict:
        """Create error result structure."""
        return {
            "property_url": property_url,
            "property_details": self._format_property_details(property_data["details"]),
            "processing_info": {
                "images_processed": 0,
                "images_analyzed": [],
                "status": "failed",
                "error_message": error_msg,
            },
            "ai_analysis_raw": None,
        }

    def _format_property_details(self, details: Dict) -> Dict:
        """Format property details for output."""
        return {
            "address": details.get("address"),
            "listed_price": details.get("price"),
            "currency": details.get("currency"),
            "bedrooms": details.get("bedrooms"),
            "bathrooms": details.get("bathrooms"),
            "property_type": details.get("property_type"),
            "mls_description": details.get("description"),
            "mls_number": details.get("mls_number"),
            "mls_is_genuine": details.get("mls_is_genuine", False),
        }

    async def process_all_properties(
        self, properties_dict: Dict[str, Dict]
    ) -> AsyncGenerator[Dict, None]:
        """Generator that processes all properties with true concurrent processing."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(client, property_url, property_data):
            async with semaphore:
                return await self.process_single_property(
                    client, property_url, property_data
                )

        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = [
                process_with_semaphore(client, property_url, property_data)
                for property_url, property_data in properties_dict.items()
            ]

            # Process all properties concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed with exception: {result}")
                    continue
                yield result

    async def _call_gemini_api(
        self, images_base64: List[str], property_details: Dict
    ) -> str:
        """Call Gemini API with retry logic, 300s timeout, and rate limiting."""
        await self._enforce_rate_limit()

        for attempt in range(self.max_retries + 1):
            try:
                # Create prompt with property details
                full_prompt = self._create_prompt_with_details(property_details)

                # Prepare content
                contents = [full_prompt]

                # Add all images as inline data
                for img_b64 in images_base64:
                    contents.append(
                        types.Part.from_bytes(
                            data=pybase64.b64decode(img_b64), mime_type="image/jpeg"
                        )
                    )

                # Call Gemini API with timeout
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, self._sync_gemini_call, contents),
                    timeout=300.0,
                )

                return response.text

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All API attempts failed: {e}")
                    raise e

    def _sync_gemini_call(self, contents):
        """Synchronous Gemini API call to be run in executor."""
        return self.client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
        )

    async def _enforce_rate_limit(self):
        """Rate limiting: 200 requests per minute (3.33 per second)."""
        async with self._rate_limit_lock:
            current_time = time.time()

            # Remove old timestamps (older than 1 minute)
            self.last_request_times = [
                t for t in self.last_request_times if current_time - t < 60
            ]

            # If we're at the rate limit, find the minimum wait time
            if len(self.last_request_times) >= self.rate_limit:
                oldest_request_time = self.last_request_times[0]
                sleep_time = 60 - (current_time - oldest_request_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.last_request_times.append(current_time)

    async def collect_and_format_results(
        self,
        results_generator: AsyncGenerator[Dict, None],
        source_url: str = None,
        removed_mls_list: List[str] = None,
    ) -> Dict:
        """Collect results from generator and format for output."""
        results = {
            "analysis_metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gemini_model_used": self.gemini_model,
                "source_data_url": source_url,
                "total_properties_processed": 0,
            },
            "properties": [],
            "removed_properties": {
                "mls_numbers": removed_mls_list or [],
                "count": len(removed_mls_list) if removed_mls_list else 0,
            },
            "processing_summary": {
                "successful_analyses": 0,
                "failed_analyses": 0,
                "total_images_processed": 0,
                "rate_limit_errors": 0,
                "genuine_mls_count": 0,
                "generated_mls_count": 0,
                "removed_properties_mls_count": len(removed_mls_list)
                if removed_mls_list
                else 0,
            },
        }

        # Collect all results from generator
        async for result in results_generator:
            results["properties"].append(result)

            # Update counters
            if result["processing_info"]["status"] == "success":
                results["processing_summary"]["successful_analyses"] += 1
            else:
                results["processing_summary"]["failed_analyses"] += 1
                if "429" in str(result["processing_info"]["error_message"]):
                    results["processing_summary"]["rate_limit_errors"] += 1

            results["processing_summary"]["total_images_processed"] += result[
                "processing_info"
            ]["images_processed"]

            # Track MLS statistics
            if result["property_details"].get("mls_is_genuine", False):
                results["processing_summary"]["genuine_mls_count"] += 1
            else:
                results["processing_summary"]["generated_mls_count"] += 1

        # Update final metadata
        results["analysis_metadata"]["total_properties_processed"] = len(
            results["properties"]
        )

        # Log final MLS statistics
        genuine_count = results["processing_summary"]["genuine_mls_count"]
        generated_count = results["processing_summary"]["generated_mls_count"]
        total_count = results["analysis_metadata"]["total_properties_processed"]
        removed_count = results["processing_summary"]["removed_properties_mls_count"]

        self.logger.info(
            f"🏷️ MLS SUMMARY: {genuine_count} genuine, {generated_count} generated out of {total_count} total properties"
        )
        self.logger.info(f"🗑️ REMOVED PROPERTIES: {removed_count} MLS numbers extracted")

        return results

    def format_removed_properties_only(
        self, removed_mls_list: List[str], source_url: str = None
    ) -> Dict:
        """Format results when only removed properties exist."""
        results = {
            "analysis_metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gemini_model_used": self.gemini_model,
                "source_data_url": source_url,
                "total_properties_processed": 0,
            },
            "properties": [],
            "removed_properties": {
                "mls_numbers": removed_mls_list,
                "count": len(removed_mls_list),
            },
            "processing_summary": {
                "successful_analyses": 0,
                "failed_analyses": 0,
                "total_images_processed": 0,
                "rate_limit_errors": 0,
                "genuine_mls_count": 0,
                "generated_mls_count": 0,
                "removed_properties_mls_count": len(removed_mls_list),
            },
        }

        self.logger.info(
            f"🗑️ REMOVED PROPERTIES ONLY: {len(removed_mls_list)} MLS numbers extracted"
        )
        return results


# BACKGROUND PROCESSING FUNCTION
@app.function(
    image=image,
    mounts=mounts,
    timeout=3600,  # 1 hour timeout
    memory=4096,  # 4GB memory
    cpu=2.0,  # 2 vCPUs
)
async def process_properties_background(s3_url: str):
    """Background function to process properties - runs asynchronously."""

    # Load environment variables at runtime
    load_dotenv("/.env")

    try:
        # Parse S3 URL (format: s3://bucket/key)
        url_parts = s3_url[5:]  # Remove 's3://'
        bucket_name = url_parts.split("/", 1)[0]
        object_key = url_parts.split("/", 1)[1]

        logger.info(
            f"Background processing started for: s3://{bucket_name}/{object_key}"
        )

        # Download JSON from S3
        import boto3

        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        data = json.loads(response["Body"].read().decode("utf-8"))

        # Process with PropertyImageProcessor
        await process_properties_async(data, bucket_name, object_key)

        logger.info(
            f"Background processing completed for: s3://{bucket_name}/{object_key}"
        )

    except Exception as e:
        logger.error(f"Background processing error for {s3_url}: {e}")


# WEB ENDPOINT - RETURNS IMMEDIATELY
@app.function(
    image=image,
    mounts=mounts,
    timeout=30,  # Short timeout since we're just validating and spawning
    memory=512,  # Less memory needed for just validation
    cpu=1.0,
)
@modal.web_endpoint(method="POST")
async def process_properties_endpoint(request_data: dict):
    """Modal web endpoint - returns immediately and processes in background."""

    try:
        # Validate input
        if not request_data or "s3_url" not in request_data:
            return {
                "error": "Missing required 's3_url' parameter",
                "status_code": 400,
            }, 400

        s3_url = request_data["s3_url"]

        # Validate S3 URL format
        if not s3_url.startswith("s3://"):
            return {
                "error": "Invalid S3 URL format. Expected: s3://bucket/key",
                "status_code": 400,
            }, 400

        url_parts = s3_url[5:]  # Remove 's3://'
        if "/" not in url_parts:
            return {
                "error": "Invalid S3 URL format. Expected: s3://bucket/key",
                "status_code": 400,
            }, 400

        # Spawn background processing task (fire-and-forget)
        process_properties_background.spawn(s3_url)

        logger.info(f"Background processing spawned for: {s3_url}")

        # Return immediately with success
        return {
            "message": "Processing started successfully",
            "s3_url": s3_url,
            "status": "processing",
            "note": "Processing will continue in background. Results will be uploaded to s3://secondbrain-after-gemini-removedmls/",
            "status_code": 200,
        }, 200

    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return {"error": f"Internal error: {str(e)}", "status_code": 500}, 500


async def process_properties_async(data: Dict, source_bucket: str, source_key: str):
    """Async processing of properties data."""
    try:
        # Load environment variables at runtime for the processing function
        load_dotenv("/.env")

        processor = PropertyImageProcessor()

        # Construct source URL for metadata
        source_url = f"s3://{source_bucket}/{source_key}"

        # Extract current properties for analysis
        properties_dict = processor.extract_properties_from_json(data)

        # Extract MLS numbers from removed properties
        removed_mls_list = processor.extract_removed_properties_mls(data)

        # Generate output filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"property_analysis_{timestamp}.json"

        if not properties_dict:
            logger.warning("No current properties to process")
            # Still save results with just removed properties if they exist
            if removed_mls_list:
                results = processor.format_removed_properties_only(
                    removed_mls_list, source_url
                )
            else:
                logger.warning("No properties or removed properties found")
                return
        else:
            # Process all current properties
            results_generator = processor.process_all_properties(properties_dict)

            # Collect and format results
            results = await processor.collect_and_format_results(
                results_generator, source_url, removed_mls_list
            )

        # Upload results to output S3 bucket
        output_bucket = "secondbrain-after-gemini-removedmls"

        try:
            import boto3

            s3_client = boto3.client("s3")
            s3_client.put_object(
                Bucket=output_bucket,
                Key=output_filename,
                Body=json.dumps(results, indent=2, ensure_ascii=False),
                ContentType="application/json",
            )

            logger.info(f"Results uploaded to s3://{output_bucket}/{output_filename}")

        except Exception as e:
            logger.error(f"Failed to upload results to S3: {e}")

    except Exception as e:
        logger.error(f"Processing error: {e}")


# No main function needed - Modal handles deployment
