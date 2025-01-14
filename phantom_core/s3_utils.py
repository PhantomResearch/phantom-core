import boto3
from botocore.exceptions import ClientError

def print_s3_bucket_size(bucket_name: str) -> None:
    """
    Print the total size of data stored in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        None

    Raises:
        ClientError: If there's an error accessing the S3 bucket.
    """
    s3 = boto3.client('s3')
    
    try:
        # Initialize total size
        total_size = 0

        # Paginate through all objects in the bucket
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                total_size += obj['Size']

        # Convert total size to appropriate unit
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(total_size)
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        print(f"Total size of data in bucket '{bucket_name}': {size:.2f} {units[unit_index]}")

    except ClientError as e:
        print(f"Error accessing S3 bucket '{bucket_name}': {e}")


def print_dvc_remote_size():
    bucket_name = 'phantom-research'
    print_s3_bucket_size(bucket_name)
