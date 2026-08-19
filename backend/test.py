import boto3
from botocore.exceptions import ClientError

ROLE_ARN = "arn:aws:iam::060730976957:role/AIAgentRole"

try:
    sts = boto3.client("sts")

    response = sts.assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName="AWSAgentTest"
    )

    print("SUCCESS!")
    print("Role assumed successfully.")
    print("Assumed role ARN:")
    print(response["AssumedRoleUser"]["Arn"])

except ClientError as e:
    print("FAILED")
    print(e)