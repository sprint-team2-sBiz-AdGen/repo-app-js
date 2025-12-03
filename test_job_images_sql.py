#!/usr/bin/env python3
"""Test script to verify image retrieval using raw SQL"""
import asyncio
import uuid
import os
from dotenv import load_dotenv
import asyncpg

# Load environment variables
load_dotenv()

# Extract database connection details from DATABASE_URL
# Format: postgresql+asyncpg://user:password@host:port/database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://feedlyai:feedlyai_dev_password_74154@localhost:5432/feedlyai")

# Parse DATABASE_URL
def parse_db_url(url):
    """Parse postgresql+asyncpg:// URL"""
    url = url.replace("postgresql+asyncpg://", "")
    parts = url.split("@")
    auth = parts[0]
    host_db = parts[1]
    user, password = auth.split(":")
    host, db = host_db.split("/")
    host_parts = host.split(":")
    hostname = host_parts[0]
    port = int(host_parts[1]) if len(host_parts) > 1 else 5432
    return user, password, hostname, port, db

async def test_job_images_sql(job_id_str: str):
    """Test retrieving images using raw SQL"""
    job_id = uuid.UUID(job_id_str)
    print(f"=" * 60)
    print(f"Testing job_id: {job_id}")
    print(f"=" * 60)
    
    # Parse database URL
    user, password, host, port, database = parse_db_url(DATABASE_URL)
    
    # Connect to database
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    
    try:
        # 1. Check if JobVariant records exist
        print("\n1. Checking JobVariant records...")
        variant_query = """
            SELECT 
                job_variants_id,
                job_id,
                img_asset_id,
                overlaid_img_asset_id,
                creation_order,
                status
            FROM jobs_variants
            WHERE job_id = $1
            ORDER BY creation_order
        """
        variants = await conn.fetch(variant_query, job_id)
        
        print(f"   Found {len(variants)} JobVariant records")
        for idx, variant in enumerate(variants, 1):
            print(f"   Variant {idx}:")
            print(f"     - job_variants_id: {variant['job_variants_id']}")
            print(f"     - overlaid_img_asset_id: {variant['overlaid_img_asset_id']}")
            print(f"     - img_asset_id: {variant['img_asset_id']}")
            print(f"     - creation_order: {variant['creation_order']}")
            print(f"     - status: {variant['status']}")
        
        # 2. Check variants with overlaid_img_asset_id
        print("\n2. Checking variants with overlaid_img_asset_id...")
        variants_with_overlay = [v for v in variants if v['overlaid_img_asset_id'] is not None]
        print(f"   Found {len(variants_with_overlay)} variants with overlaid_img_asset_id")
        
        # 3. Execute the same query as the endpoint (JOIN query)
        print("\n3. Executing image query (same as endpoint - JOIN query)...")
        image_query = """
            SELECT 
                ia.image_url
            FROM image_assets ia
            INNER JOIN jobs_variants jv ON jv.overlaid_img_asset_id = ia.image_asset_id
            WHERE jv.job_id = $1
            ORDER BY jv.creation_order
            LIMIT 3
        """
        images = await conn.fetch(image_query, job_id)
        
        print(f"   Found {len(images)} images:")
        for idx, img in enumerate(images, 1):
            print(f"   Image {idx}: {img['image_url']}")
        
        # 4. Get detailed image information
        print("\n4. Getting detailed image information...")
        detailed_query = """
            SELECT 
                ia.image_asset_id,
                ia.image_url,
                jv.creation_order
            FROM image_assets ia
            INNER JOIN jobs_variants jv ON jv.overlaid_img_asset_id = ia.image_asset_id
            WHERE jv.job_id = $1
            ORDER BY jv.creation_order
            LIMIT 3
        """
        detailed_images = await conn.fetch(detailed_query, job_id)
        
        print(f"   Detailed images ({len(detailed_images)}):")
        for idx, img in enumerate(detailed_images, 1):
            print(f"   Image {idx} (order {img['creation_order']}):")
            print(f"     - image_asset_id: {img['image_asset_id']}")
            print(f"     - image_url: {img['image_url']}")
        
        # 5. Check if image_assets exist for the overlaid_img_asset_ids
        print("\n5. Verifying image_assets exist for overlaid_img_asset_ids...")
        if variants_with_overlay:
            overlay_ids = [v['overlaid_img_asset_id'] for v in variants_with_overlay]
            asset_check_query = """
                SELECT 
                    image_asset_id,
                    image_url,
                    image_type
                FROM image_assets
                WHERE image_asset_id = ANY($1::uuid[])
            """
            assets = await conn.fetch(asset_check_query, overlay_ids)
            print(f"   Found {len(assets)} matching image_assets:")
            for asset in assets:
                print(f"     - image_asset_id: {asset['image_asset_id']}")
                print(f"     - image_url: {asset['image_url']}")
                print(f"     - image_type: {asset['image_type']}")
        
        print("\n" + "=" * 60)
        if len(images) > 0:
            print(f"✅ SUCCESS: Retrieved {len(images)} images")
        else:
            print(f"❌ WARNING: No images found")
            if len(variants) == 0:
                print("   Reason: No JobVariant records found for this job_id")
            elif len(variants_with_overlay) == 0:
                print("   Reason: JobVariant records exist but overlaid_img_asset_id is NULL")
            else:
                print("   Reason: Images might not be linked correctly")
        print("=" * 60)
        
    finally:
        await conn.close()

if __name__ == "__main__":
    job_id = "b02650e3-c70d-40a0-ad86-758aee877682"
    asyncio.run(test_job_images_sql(job_id))

