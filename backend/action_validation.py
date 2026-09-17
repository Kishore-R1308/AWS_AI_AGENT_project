from typing import Any, Dict, Tuple


ALLOWED_ACTIONS = {
    "create",
    "start",
    "stop",
    "reboot",
    "enable",
    "disable",
    "delete",
}


RESOURCE_REQUIRED_FIELDS = {
    "s3_bucket": {
        "create": ["bucket_name"],
        "delete": ["bucket_name", "delete_confirmation"],
    },
    "ec2_instance": {
        "start": ["instance_id"],
        "stop": ["instance_id"],
        "reboot": ["instance_id"],
        "create": [
            "ami_id",
            "instance_type",
            "key_name",
        ],
        "delete": [
            "instance_id",
            "delete_confirmation",
        ],
    },
    "rds_instance": {
        "start": ["db_instance_identifier"],
        "stop": ["db_instance_identifier"],
        "create": [
            "db_instance_identifier",
            "db_instance_class",
            "engine",
            "master_username",
            "master_password",
        ],
        "delete": [
            "db_instance_identifier",
            "delete_confirmation",
        ],
    },
    "lambda_function": {
        "enable": ["function_name"],
        "disable": ["function_name"],
        "create": [
            "function_name",
            "runtime",
            "role_arn",
            "handler",
            "zip_file",
        ],
        "delete": [
            "function_name",
            "delete_confirmation",
        ],
    },
    "security_group": {
        "create": [
            "group_name",
            "description",
            "vpc_id",
        ],
        "delete": [
            "group_id",
            "delete_confirmation",
        ],
    },
    "vpc": {
        "create": [
            "cidr_block",
        ],
        "delete": [
            "vpc_id",
            "delete_confirmation",
        ],
    },
    "subnet": {
        "create": [
            "vpc_id",
            "cidr_block",
            "availability_zone",
        ],
        "delete": [
            "subnet_id",
            "delete_confirmation",
        ],
    },
    "iam_resource": {
        "create": [
            "resource_name",
            "resource_kind",
        ],
        "delete": [
            "resource_name",
            "resource_kind",
            "delete_confirmation",
        ],
    },
}


def validate_action(
    action: str,
    resource_type: str,
    parameters: Dict[str, Any],
) -> Tuple[bool, str]:

    if action not in ALLOWED_ACTIONS:
        return False, f"Unsupported action: {action}"

    if resource_type not in RESOURCE_REQUIRED_FIELDS:
        return False, (
            f"Unsupported resource type: {resource_type}"
        )

    if not isinstance(parameters, dict):
        return False, (
            "Parameters must be a dictionary"
        )

    resource_actions = RESOURCE_REQUIRED_FIELDS[resource_type]
    if action not in resource_actions:
        return False, (
            f"Action '{action}' is not supported for resource type "
            f"'{resource_type}'"
        )

    required_fields = resource_actions[action]

    missing_fields = [
        field
        for field in required_fields
        if field not in parameters
        or parameters[field] in (None, "")
    ]

    if missing_fields:
        return False, (
            "Missing required fields: "
            + ", ".join(missing_fields)
        )

    if action == "delete":
        resource_identifier = parameters.get(
            "delete_confirmation"
        )

        resource_name = (
            parameters.get("bucket_name")
            or parameters.get("instance_id")
            or parameters.get(
                "db_instance_identifier"
            )
            or parameters.get("function_name")
            or parameters.get("group_id")
            or parameters.get("vpc_id")
            or parameters.get("subnet_id")
            or parameters.get("resource_name")
        )

        if resource_identifier != resource_name:
            return False, (
                "delete_confirmation must exactly "
                "match the resource identifier"
            )

    return True, "Action is valid"