"""
AWS Action Executor

Supported services:
1. S3 Bucket
2. EC2 Instance
3. RDS Instance
4. Lambda Function
5. Security Group
6. VPC
7. Subnet
8. IAM Resources

Only explicitly supported operations are allowed.
User confirmation must be handled by the API layer
before this executor is called.
"""

from typing import Any, Dict, List, Optional
import base64
import json
import re

from aws_auth import get_aws_client


SUPPORTED_ACTIONS = {
    "create": {
        "s3_bucket",
        "ec2_instance",
        "rds_instance",
        "lambda_function",
        "security_group",
        "vpc",
        "subnet",
        "iam_resource",
    },
    "start": {"ec2_instance", "rds_instance"},
    "stop": {"ec2_instance", "rds_instance"},
    "reboot": {"ec2_instance"},
    "enable": {"lambda_function"},
    "disable": {"lambda_function"},
    "delete": {
        "s3_bucket",
        "ec2_instance",
        "rds_instance",
        "lambda_function",
        "security_group",
        "vpc",
        "subnet",
        "iam_resource",
    },
}


def execute_aws_action(
    session_id: str,
    action: str,
    resource_type: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute an approved AWS operation.

    The API must validate confirmation before calling
    this function.
    """

    if action not in SUPPORTED_ACTIONS:
        raise ValueError(
            f"Unsupported action: {action}"
        )

    if resource_type not in SUPPORTED_ACTIONS[action]:
        raise ValueError(
            f"Unsupported resource type: {resource_type}"
        )

    if not isinstance(parameters, dict):
        raise ValueError(
            "parameters must be a dictionary"
        )

    executors = {
        "s3_bucket": execute_s3,
        "ec2_instance": execute_ec2,
        "rds_instance": execute_rds,
        "lambda_function": execute_lambda,
        "security_group": execute_security_group,
        "vpc": execute_vpc,
        "subnet": execute_subnet,
        "iam_resource": execute_iam,
    }

    executor = executors.get(resource_type)

    if executor is None:
        raise ValueError(
            f"No executor for {resource_type}"
        )

    return executor(
        session_id=session_id,
        action=action,
        parameters=parameters,
    )


def require_string(
    parameters: Dict[str, Any],
    field: str,
) -> str:
    """
    Validate and return a required string.
    """

    value = parameters.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field} must be a non-empty string"
        )

    return value.strip()


def validate_identifier(
    value: str,
    field: str,
) -> str:
    """
    Basic identifier validation.
    """

    if len(value) > 255:
        raise ValueError(
            f"{field} is too long"
        )

    return value


def require_delete_confirmation(
    parameters: Dict[str, Any],
    resource_identifier: str,
) -> None:
    """
    Require exact resource identifier confirmation.
    """

    confirmation = parameters.get(
        "delete_confirmation"
    )

    if confirmation != resource_identifier:
        raise ValueError(
            "delete_confirmation must exactly match "
            "the resource identifier"
        )


def get_region(
    parameters: Dict[str, Any],
) -> str:
    """
    Get the requested AWS region.
    """

    region = parameters.get(
        "region",
        "us-east-1",
    )

    if not isinstance(region, str) or not region:
        raise ValueError(
            "region must be a valid string"
        )

    return region


# =====================================================
# 1. S3
# =====================================================


def execute_s3(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="s3",
    )

    bucket_name = validate_identifier(
        require_string(parameters, "bucket_name"),
        "bucket_name",
    )

    if action == "create":

        region = get_region(parameters)

        if region == "us-east-1":
            response = client.create_bucket(
                Bucket=bucket_name,
            )
        else:
            response = client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": region,
                },
            )

        return {
            "success": True,
            "service": "s3",
            "action": "create",
            "bucket_name": bucket_name,
            "response_metadata": response.get(
                "ResponseMetadata",
                {},
            ),
        }

    if action == "delete":

        require_delete_confirmation(
            parameters,
            bucket_name,
        )

        client.delete_bucket(
            Bucket=bucket_name,
        )

        return {
            "success": True,
            "service": "s3",
            "action": "delete",
            "bucket_name": bucket_name,
        }

    raise ValueError(
        f"Unsupported S3 action: {action}"
    )


# =====================================================
# 2. EC2
# =====================================================


def execute_ec2(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="ec2",
    )

    if action == "create":

        ami_id = require_string(
            parameters,
            "ami_id",
        )

        instance_type = require_string(
            parameters,
            "instance_type",
        )

        key_name = require_string(
            parameters,
            "key_name",
        )

        request = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "KeyName": key_name,
            "MinCount": 1,
            "MaxCount": 1,
        }

        subnet_id = parameters.get("subnet_id")

        if subnet_id:
            request["SubnetId"] = subnet_id

        security_group_ids = parameters.get(
            "security_group_ids"
        )

        if security_group_ids:
            if not isinstance(
                security_group_ids,
                list,
            ):
                raise ValueError(
                    "security_group_ids must be a list"
                )

            request["SecurityGroupIds"] = (
                security_group_ids
            )

        response = client.run_instances(
            **request
        )

        instance = response["Instances"][0]

        return {
            "success": True,
            "service": "ec2",
            "action": "create",
            "instance_id": instance.get(
                "InstanceId"
            ),
            "message": (
                "EC2 instance launch request submitted"
            ),
        }

    if action in {"start", "stop", "reboot"}:

        instance_id = require_string(
            parameters,
            "instance_id",
        )

        response = client.describe_instances(
            InstanceIds=[instance_id],
        )
        reservations = response.get("Reservations", [])
        instances = [
            instance
            for reservation in reservations
            for instance in reservation.get("Instances", [])
        ]
        if not instances:
            raise ValueError(f"EC2 instance not found: {instance_id}")

        current_state = instances[0].get("State", {}).get("Name")
        if action == "start":
            if current_state != "stopped":
                raise ValueError(
                    f"Instance {instance_id} is {current_state}; only stopped instances can be started"
                )
            response = client.start_instances(InstanceIds=[instance_id])
            message = "EC2 start request submitted"
        elif action == "stop":
            if current_state != "running":
                raise ValueError(
                    f"Instance {instance_id} is {current_state}; only running instances can be stopped"
                )
            response = client.stop_instances(InstanceIds=[instance_id])
            message = "EC2 stop request submitted"
        else:
            if current_state != "running":
                raise ValueError(
                    f"Instance {instance_id} is {current_state}; only running instances can be rebooted"
                )
            response = client.reboot_instances(InstanceIds=[instance_id])
            message = "EC2 reboot request submitted"

        return {
            "success": True,
            "service": "ec2",
            "action": action,
            "instance_id": instance_id,
            "previous_state": current_state,
            "message": message,
            "response_metadata": response.get("ResponseMetadata", {}),
        }

    if action == "delete":

        instance_id = require_string(
            parameters,
            "instance_id",
        )

        require_delete_confirmation(
            parameters,
            instance_id,
        )

        response = client.terminate_instances(
            InstanceIds=[instance_id],
        )

        return {
            "success": True,
            "service": "ec2",
            "action": "delete",
            "instance_id": instance_id,
            "message": (
                "EC2 termination request submitted"
            ),
            "response_metadata": response.get(
                "ResponseMetadata",
                {},
            ),
        }

    raise ValueError(
        f"Unsupported EC2 action: {action}"
    )


# =====================================================
# 3. RDS
# =====================================================


def execute_rds(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="rds",
    )

    identifier = require_string(
        parameters,
        "db_instance_identifier",
    )

    if action == "create":

        engine = require_string(
            parameters,
            "engine",
        )

        instance_class = require_string(
            parameters,
            "db_instance_class",
        )

        master_username = require_string(
            parameters,
            "master_username",
        )

        master_password = require_string(
            parameters,
            "master_password",
        )

        allocated_storage = parameters.get(
            "allocated_storage",
            20,
        )

        request = {
            "DBInstanceIdentifier": identifier,
            "DBInstanceClass": instance_class,
            "Engine": engine,
            "MasterUsername": master_username,
            "MasterUserPassword": master_password,
            "AllocatedStorage": int(
                allocated_storage
            ),
        }

        database_name = parameters.get(
            "database_name"
        )

        if database_name:
            request["DBName"] = database_name

        publicly_accessible = parameters.get(
            "publicly_accessible",
            False,
        )

        request["PubliclyAccessible"] = bool(
            publicly_accessible
        )

        response = client.create_db_instance(
            **request
        )

        return {
            "success": True,
            "service": "rds",
            "action": "create",
            "db_instance_identifier": identifier,
            "message": (
                "RDS creation request submitted"
            ),
            "response_metadata": response.get(
                "ResponseMetadata",
                {},
            ),
        }

    if action in {"start", "stop"}:
        if action == "start":
            response = client.start_db_instance(DBInstanceIdentifier=identifier)
            message = "RDS start request submitted"
        else:
            response = client.stop_db_instance(DBInstanceIdentifier=identifier)
            message = "RDS stop request submitted"

        return {
            "success": True,
            "service": "rds",
            "action": action,
            "db_instance_identifier": identifier,
            "message": message,
            "response_metadata": response.get("ResponseMetadata", {}),
        }

    if action == "delete":

        require_delete_confirmation(
            parameters,
            identifier,
        )

        skip_final_snapshot = parameters.get(
            "skip_final_snapshot",
            False,
        )

        response = client.delete_db_instance(
            DBInstanceIdentifier=identifier,
            SkipFinalSnapshot=bool(
                skip_final_snapshot
            ),
        )

        return {
            "success": True,
            "service": "rds",
            "action": "delete",
            "db_instance_identifier": identifier,
            "message": (
                "RDS deletion request submitted"
            ),
            "response_metadata": response.get(
                "ResponseMetadata",
                {},
            ),
        }

    raise ValueError(
        f"Unsupported RDS action: {action}"
    )


# =====================================================
# 4. Lambda
# =====================================================


def execute_lambda(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="lambda",
    )

    function_name = require_string(
        parameters,
        "function_name",
    )

    if action == "create":

        runtime = require_string(
            parameters,
            "runtime",
        )

        role_arn = require_string(
            parameters,
            "role_arn",
        )

        handler = require_string(
            parameters,
            "handler",
        )

        zip_file = parameters.get("zip_file")

        if not isinstance(zip_file, str):
            raise ValueError(
                "zip_file must be a base64-encoded string"
            )

        try:
            code_bytes = base64.b64decode(
                zip_file,
                validate=True,
            )
        except Exception as exc:
            raise ValueError(
                "zip_file is not valid base64"
            ) from exc

        response = client.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role_arn,
            Handler=handler,
            Code={
                "ZipFile": code_bytes,
            },
            Publish=True,
        )

        return {
            "success": True,
            "service": "lambda",
            "action": "create",
            "function_name": function_name,
            "message": (
                "Lambda function created successfully"
            ),
            "response_metadata": response.get(
                "ResponseMetadata",
                {},
            ),
        }

    if action == "disable":
        response = client.put_function_concurrency(
            FunctionName=function_name,
            ReservedConcurrentExecutions=0,
        )
        return {
            "success": True,
            "service": "lambda",
            "action": "disable",
            "function_name": function_name,
            "message": "Lambda function disabled by setting reserved concurrency to 0",
            "response_metadata": response.get("ResponseMetadata", {}),
        }

    if action == "enable":
        response = client.delete_function_concurrency(
            FunctionName=function_name,
        )
        return {
            "success": True,
            "service": "lambda",
            "action": "enable",
            "function_name": function_name,
            "message": "Lambda function enabled by removing reserved concurrency limit",
            "response_metadata": response.get("ResponseMetadata", {}),
        }

    if action == "delete":

        require_delete_confirmation(
            parameters,
            function_name,
        )

        client.delete_function(
            FunctionName=function_name,
        )

        return {
            "success": True,
            "service": "lambda",
            "action": "delete",
            "function_name": function_name,
        }

    raise ValueError(
        f"Unsupported Lambda action: {action}"
    )


# =====================================================
# 5. Security Group
# =====================================================


def execute_security_group(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="ec2",
    )

    if action == "create":

        group_name = require_string(
            parameters,
            "group_name",
        )

        description = require_string(
            parameters,
            "description",
        )

        vpc_id = require_string(
            parameters,
            "vpc_id",
        )

        response = client.create_security_group(
            GroupName=group_name,
            Description=description,
            VpcId=vpc_id,
        )

        return {
            "success": True,
            "service": "security_group",
            "action": "create",
            "group_id": response.get(
                "GroupId"
            ),
            "group_name": group_name,
        }

    if action == "delete":

        group_id = require_string(
            parameters,
            "group_id",
        )

        require_delete_confirmation(
            parameters,
            group_id,
        )

        client.delete_security_group(
            GroupId=group_id,
        )

        return {
            "success": True,
            "service": "security_group",
            "action": "delete",
            "group_id": group_id,
        }

    raise ValueError(
        f"Unsupported Security Group action: {action}"
    )


# =====================================================
# 6. VPC
# =====================================================


def execute_vpc(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="ec2",
    )

    if action == "create":

        cidr_block = require_string(
            parameters,
            "cidr_block",
        )

        response = client.create_vpc(
            CidrBlock=cidr_block,
        )

        vpc = response.get(
            "Vpc",
            {},
        )

        return {
            "success": True,
            "service": "vpc",
            "action": "create",
            "vpc_id": vpc.get("VpcId"),
            "cidr_block": cidr_block,
        }

    if action == "delete":

        vpc_id = require_string(
            parameters,
            "vpc_id",
        )

        require_delete_confirmation(
            parameters,
            vpc_id,
        )

        client.delete_vpc(
            VpcId=vpc_id,
        )

        return {
            "success": True,
            "service": "vpc",
            "action": "delete",
            "vpc_id": vpc_id,
        }

    raise ValueError(
        f"Unsupported VPC action: {action}"
    )


# =====================================================
# 7. Subnet
# =====================================================


def execute_subnet(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="ec2",
    )

    if action == "create":

        vpc_id = require_string(
            parameters,
            "vpc_id",
        )

        cidr_block = require_string(
            parameters,
            "cidr_block",
        )

        availability_zone = require_string(
            parameters,
            "availability_zone",
        )

        response = client.create_subnet(
            VpcId=vpc_id,
            CidrBlock=cidr_block,
            AvailabilityZone=availability_zone,
        )

        subnet = response.get(
            "Subnet",
            {},
        )

        return {
            "success": True,
            "service": "subnet",
            "action": "create",
            "subnet_id": subnet.get(
                "SubnetId"
            ),
            "vpc_id": vpc_id,
        }

    if action == "delete":

        subnet_id = require_string(
            parameters,
            "subnet_id",
        )

        require_delete_confirmation(
            parameters,
            subnet_id,
        )

        client.delete_subnet(
            SubnetId=subnet_id,
        )

        return {
            "success": True,
            "service": "subnet",
            "action": "delete",
            "subnet_id": subnet_id,
        }

    raise ValueError(
        f"Unsupported Subnet action: {action}"
    )


# =====================================================
# 8. IAM
# =====================================================


def execute_iam(
    session_id: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:

    client = get_aws_client(
        session_id=session_id,
        service_name="iam",
    )

    resource_kind = require_string(
        parameters,
        "resource_kind",
    )

    resource_name = require_string(
        parameters,
        "resource_name",
    )

    supported_kinds = {
        "role",
        "user",
        "group",
    }

    if resource_kind not in supported_kinds:
        raise ValueError(
            "Supported IAM resource_kind values: "
            "role, user, group"
        )

    if action == "create":

        if resource_kind == "role":

            assume_role_policy = parameters.get(
                "assume_role_policy"
            )

            if not assume_role_policy:
                raise ValueError(
                    "assume_role_policy is required "
                    "for IAM roles"
                )

            if isinstance(
                assume_role_policy,
                dict,
            ):
                assume_role_policy = json.dumps(
                    assume_role_policy
                )

            response = client.create_role(
                RoleName=resource_name,
                AssumeRolePolicyDocument=(
                    assume_role_policy
                ),
            )

            return {
                "success": True,
                "service": "iam",
                "resource_kind": "role",
                "action": "create",
                "resource_name": resource_name,
                "arn": response["Role"].get(
                    "Arn"
                ),
            }

        if resource_kind == "user":

            response = client.create_user(
                UserName=resource_name,
            )

            return {
                "success": True,
                "service": "iam",
                "resource_kind": "user",
                "action": "create",
                "resource_name": resource_name,
                "arn": response["User"].get(
                    "Arn"
                ),
            }

        if resource_kind == "group":

            response = client.create_group(
                GroupName=resource_name,
            )

            return {
                "success": True,
                "service": "iam",
                "resource_kind": "group",
                "action": "create",
                "resource_name": resource_name,
                "arn": response["Group"].get(
                    "Arn"
                ),
            }

    if action == "delete":

        require_delete_confirmation(
            parameters,
            resource_name,
        )

        if resource_kind == "role":

            client.delete_role(
                RoleName=resource_name,
            )

        elif resource_kind == "user":

            client.delete_user(
                UserName=resource_name,
            )

        elif resource_kind == "group":

            client.delete_group(
                GroupName=resource_name,
            )

        return {
            "success": True,
            "service": "iam",
            "resource_kind": resource_kind,
            "action": "delete",
            "resource_name": resource_name,
        }

    raise ValueError(
        f"Unsupported IAM action: {action}"
    )