from datetime import datetime, timedelta, timezone, date
from aws_auth import get_aws_client
 
 
 
def get_ec2_instances(session_id: str):
 
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_instances()
 
    results = []
 
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
 
            name = "Unnamed"
 
            for tag in instance.get("Tags", []):
                if tag.get("Key") == "Name":
                    name = tag.get("Value")
 
            results.append(
                {
                    "instance_id": instance.get("InstanceId"),
                    "name": name,
                    "state": instance.get("State", {}).get("Name"),
                    "instance_type": instance.get("InstanceType"),
                    "private_ip": instance.get("PrivateIpAddress"),
                    "public_ip": instance.get("PublicIpAddress"),
                }
            )
 
    return results
 
 
 
def get_s3_buckets(session_id: str):
 
    s3 = get_aws_client(session_id, "s3")
    response = s3.list_buckets()
 
    return [
        {
            "name": bucket["Name"],
            "created": str(bucket.get("CreationDate")),
        }
        for bucket in response.get("Buckets", [])
    ]
 
 
 
def get_rds_instances(session_id: str):
 
    rds = get_aws_client(session_id, "rds")
    response = rds.describe_db_instances()
 
    return [
        {
            "identifier": db.get("DBInstanceIdentifier"),
            "status": db.get("DBInstanceStatus"),
            "engine": db.get("Engine"),
            "engine_version": db.get("EngineVersion"),
            "instance_class": db.get("DBInstanceClass"),
            "storage_gb": db.get("AllocatedStorage"),
        }
        for db in response.get("DBInstances", [])
    ]
 
 
 
def get_s3_storage_summary(session_id: str):
 
    s3 = get_aws_client(session_id, "s3")
    response = s3.list_buckets()
 
    results = []
 
    for bucket in response.get("Buckets", []):
 
        bucket_name = bucket["Name"]
        total_bytes = 0
        object_count = 0
 
        try:
            paginator = s3.get_paginator("list_objects_v2")
 
            for page in paginator.paginate(Bucket=bucket_name):
                for obj in page.get("Contents", []):
 
                    total_bytes += obj.get("Size", 0)
                    object_count += 1
 
            results.append(
                {
                    "bucket": bucket_name,
                    "object_count": object_count,
                    "size_bytes": total_bytes,
                    "size_mb": round(
                        total_bytes / (1024 * 1024),
                        2,
                    ),
                    "size_gb": round(
                        total_bytes / (1024 * 1024 * 1024),
                        4,
                    ),
                }
            )
 
        except Exception as exc:
            results.append(
                {
                    "bucket": bucket_name,
                    "error": str(exc),
                }
            )
 
    results.sort(
        key=lambda item: item.get(
            "size_bytes",
            0,
        ),
        reverse=True,
    )
 
    return results
 
 
 
def get_vpcs(session_id: str):
 
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_vpcs()
 
    results = []
 
    for vpc in response.get("Vpcs", []):
 
        name = "Unnamed"
 
        for tag in vpc.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")
 
        results.append(
            {
                "vpc_id": vpc.get("VpcId"),
                "name": name,
                "cidr_block": vpc.get("CidrBlock"),
                "state": vpc.get("State"),
                "is_default": vpc.get("IsDefault"),
            }
        )
 
    return results
 
 
 
def get_subnets(session_id: str):
 
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_subnets()
 
    results = []
 
    for subnet in response.get("Subnets", []):
 
        name = "Unnamed"
 
        for tag in subnet.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")
 
        results.append(
            {
                "subnet_id": subnet.get("SubnetId"),
                "name": name,
                "vpc_id": subnet.get("VpcId"),
                "cidr_block": subnet.get("CidrBlock"),
                "availability_zone": subnet.get(
                    "AvailabilityZone"
                ),
            }
        )
 
    return results
 
 
 
def get_internet_gateways(session_id: str):
 
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_internet_gateways()
 
    results = []
 
    for gateway in response.get(
        "InternetGateways",
        [],
    ):
 
        name = "Unnamed"
 
        for tag in gateway.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")
 
        vpc_ids = []
 
        for attachment in gateway.get(
            "Attachments",
            [],
        ):
            vpc_ids.append(
                attachment.get("VpcId")
            )
 
        results.append(
            {
                "internet_gateway_id": gateway.get(
                    "InternetGatewayId"
                ),
                "name": name,
                "attached_vpcs": vpc_ids,
            }
        )
 
    return results
 
 
 
def get_cost_summary(session_id: str):
 
    ce = get_aws_client(session_id, "ce")
 
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
 
    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )
 
    return response.get(
        "ResultsByTime",
        []
    )
 
 
def get_cost_by_service(session_id: str):
 
    ce = get_aws_client(session_id, "ce")
 
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
 
    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.strftime("%Y-%m-%d"),
            "End": end_date.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[
            {
                "Type": "DIMENSION",
                "Key": "SERVICE",
            }
        ],
    )
 
    return response.get(
        "ResultsByTime",
        []
    )
 
 
def get_patch_status(session_id: str):
 
    ssm = get_aws_client(session_id, "ssm")
 
    managed_instances = (
        ssm.describe_instance_information()
    )
 
    instance_ids = [
        item["InstanceId"]
        for item in managed_instances.get(
            "InstanceInformationList",
            []
        )
    ]
 
    if not instance_ids:
        return []
 
    response = ssm.describe_instance_patch_states(
        InstanceIds=instance_ids
    )
 
    return [
        {
            "instance_id": item.get(
                "InstanceId"
            ),
            "missing_count": item.get(
                "MissingCount"
            ),
            "installed_count": item.get(
                "InstalledCount"
            ),
            "failed_count": item.get(
                "FailedCount"
            ),
            "operation_end_time": str(
                item.get(
                    "OperationEndTime"
                )
            ),
        }
        for item in response.get(
            "InstancePatchStates",
            []
        )
    ]
 
def get_lambda_functions(session_id):
    lambda_client = get_aws_client(session_id, "lambda")
 
    try:
        paginator = lambda_client.get_paginator("list_functions")
        functions = []
 
        for page in paginator.paginate():
            for fn in page.get("Functions", []):
                functions.append({
                    "function_name": fn.get("FunctionName"),
                    "runtime": fn.get("Runtime"),
                    "last_modified": fn.get("LastModified"),
                    "arn": fn.get("FunctionArn"),
                })
 
        return functions
 
    except Exception as e:
        return {"error": f"Failed to list Lambda functions: {str(e)}"}
 
 
def get_cloudwatch_metrics(session_id, instance_id=None):
    cloudwatch = get_aws_client(session_id, "cloudwatch")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)  # wider window

    # Standard EC2 metrics available without the CloudWatch agent
    METRICS = [
        ("CPUUtilization", "Average"),
        ("NetworkIn", "Sum"),
        ("NetworkOut", "Sum"),
        ("NetworkPacketsIn", "Sum"),
        ("NetworkPacketsOut", "Sum"),
        ("DiskReadBytes", "Sum"),
        ("DiskWriteBytes", "Sum"),
        ("DiskReadOps", "Sum"),
        ("DiskWriteOps", "Sum"),
        ("StatusCheckFailed", "Maximum"),
        ("StatusCheckFailed_Instance", "Maximum"),
        ("StatusCheckFailed_System", "Maximum"),
    ]

    def sanitize_id(instance_id, metric_name, idx):
        # CloudWatch Id must match ^[a-z][a-zA-Z0-9_]*$
        clean_instance = instance_id.replace("-", "_")
        clean_metric = metric_name.lower()
        return f"m_{clean_instance}{clean_metric}{idx}"

    if instance_id:
        target_instances = [{"instance_id": instance_id}]
    else:
        target_instances = get_ec2_instances(session_id)

    queries = []

    for idx, inst in enumerate(target_instances):
        for metric_name, stat in METRICS:
            queries.append({
                "Id": sanitize_id(
                    inst["instance_id"],
                    metric_name,
                    idx
                ),
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/EC2",
                        "MetricName": metric_name,
                        "Dimensions": [
                            {
                                "Name": "InstanceId",
                                "Value": inst["instance_id"]
                            }
                        ]
                    },
                    "Period": 300,
                    "Stat": stat
                },
                "ReturnData": True,
                "Label": f"{inst['instance_id']} - {metric_name}"
            })

    if not queries:
        return []

    # CloudWatch caps get_metric_data at 500 queries per call
    all_results = []

    for i in range(0, len(queries), 500):
        batch = queries[i:i + 500]

        response = cloudwatch.get_metric_data(
            MetricDataQueries=batch,
            StartTime=start_time,
            EndTime=end_time
        )

        all_results.extend(
            response.get("MetricDataResults", [])
        )

    return all_results

def get_cloudtrail_events(session_id):
    cloudtrail = get_aws_client(session_id, "cloudtrail")
 
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)
 
    response = cloudtrail.lookup_events(
        StartTime=start_time,
        EndTime=end_time,
        MaxResults=50
    )
 
    results = []
 
    for event in response.get("Events", []):
        results.append({
            "event_name": event.get("EventName"),
            "event_time": str(event.get("EventTime")),
            "username": event.get("Username"),
            "event_id": event.get("EventId"),
            "resources": event.get("Resources", [])
        })
 
    return results 
 
def get_resource_tags(session_id):
 
    tagging = get_aws_client(
        session_id,
        "resourcegroupstaggingapi"
    )
 
    results = []
 
    paginator = tagging.get_paginator(
        "get_resources"
    )
 
    for page in paginator.paginate():
 
        for resource in page.get(
            "ResourceTagMappingList",
            []
        ):
 
            results.append({
                "resource_arn":
                    resource.get(
                        "ResourceARN"
                    ),
                "tags":
                    resource.get(
                        "Tags",
                        []
                    )
            })
 
    return results
 
 
 
def get_ec2_tags(session_id):
 
    ec2 = get_aws_client(
        session_id,
        "ec2"
    )
 
    response = ec2.describe_instances()
 
    results = []
 
    for reservation in response.get(
        "Reservations",
        []
    ):
 
        for instance in reservation.get(
            "Instances",
            []
        ):
 
            tags = {}
 
            for tag in instance.get(
                "Tags",
                []
            ):
 
                tags[tag["Key"]] = tag["Value"]
 
            results.append({
                "instance_id":
                    instance["InstanceId"],
                "tags":
                    tags
            })
 
    return results
 
 
 
def get_s3_tags(session_id):
 
    s3 = get_aws_client(
        session_id,
        "s3"
    )
 
    buckets = s3.list_buckets()
 
    results = []
 
    for bucket in buckets.get(
        "Buckets",
        []
    ):
 
        bucket_name = bucket["Name"]
 
        try:
 
            response = s3.get_bucket_tagging(
                Bucket=bucket_name
            )
 
            tags = {
                tag["Key"]: tag["Value"]
                for tag in response.get(
                    "TagSet",
                    []
                )
            }
 
        except Exception:
            tags = {}
 
        results.append({
            "bucket": bucket_name,
            "tags": tags
        })
 
    return results
 
 
 
def get_lambda_tags(session_id):
 
    lambda_client = get_aws_client(
        session_id,
        "lambda"
    )
 
    paginator = lambda_client.get_paginator(
        "list_functions"
    )
 
    results = []
 
    for page in paginator.paginate():
 
        for fn in page.get(
            "Functions",
            []
        ):
 
            try:
 
                tags = lambda_client.list_tags(
                    Resource=fn["FunctionArn"]
                )
 
                results.append({
                    "function_name":
                        fn["FunctionName"],
                    "tags":
                        tags.get(
                            "Tags",
                            {}
                        )
                })
 
            except Exception as e:
 
                results.append({
                    "function_name":
                        fn["FunctionName"],
                    "error":
                        str(e)
                })
 
    return results


def get_route_tables(session_id: str):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_route_tables()

    results = []

    for rt in response.get("RouteTables", []):
        name = "Unnamed"
        for tag in rt.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value")

        routes = []
        for route in rt.get("Routes", []):
            routes.append({
                "destination_cidr_block": route.get("DestinationCidrBlock"),
                "gateway_id": route.get("GatewayId"),
                "instance_id": route.get("InstanceId"),
                "nat_gateway_id": route.get("NatGatewayId"),
                "transit_gateway_id": route.get("TransitGatewayId"),
                "state": route.get("State"),
            })

        associations = []
        for assoc in rt.get("Associations", []):
            associations.append({
                "subnet_id": assoc.get("SubnetId"),
                "main": assoc.get("Main"),
                "route_table_association_id": assoc.get("RouteTableAssociationId"),
            })

        results.append({
            "route_table_id": rt.get("RouteTableId"),
            "name": name,
            "vpc_id": rt.get("VpcId"),
            "routes": routes,
            "associations": associations,
        })

    return results


def get_security_groups(session_id: str):
    ec2 = get_aws_client(session_id, "ec2")
    response = ec2.describe_security_groups()

    results = []

    for sg in response.get("SecurityGroups", []):
        name = sg.get("GroupName", "Unnamed")

        inbound_rules = []
        for rule in sg.get("IpPermissions", []):
            inbound_rules.append({
                "protocol": rule.get("IpProtocol"),
                "from_port": rule.get("FromPort"),
                "to_port": rule.get("ToPort"),
                "ip_ranges": [ip.get("CidrIp") for ip in rule.get("IpRanges", [])],
                "ipv6_ranges": [ip.get("CidrIpv6") for ip in rule.get("Ipv6Ranges", [])],
                "user_id_group_pairs": [pair.get("GroupId") for pair in rule.get("UserIdGroupPairs", [])],
            })

        outbound_rules = []
        for rule in sg.get("IpPermissionsEgress", []):
            outbound_rules.append({
                "protocol": rule.get("IpProtocol"),
                "from_port": rule.get("FromPort"),
                "to_port": rule.get("ToPort"),
                "ip_ranges": [ip.get("CidrIp") for ip in rule.get("IpRanges", [])],
                "ipv6_ranges": [ip.get("CidrIpv6") for ip in rule.get("Ipv6Ranges", [])],
                "user_id_group_pairs": [pair.get("GroupId") for pair in rule.get("UserIdGroupPairs", [])],
            })

        results.append({
            "group_id": sg.get("GroupId"),
            "name": name,
            "description": sg.get("Description"),
            "vpc_id": sg.get("VpcId"),
            "inbound_rules": inbound_rules,
            "outbound_rules": outbound_rules,
        })

    return results


def get_cloudwatch_alarms(session_id):
    cloudwatch = get_aws_client(session_id, "cloudwatch")
 
    response = cloudwatch.describe_alarms()
 
    return response.get("MetricAlarms", [])
 
def get_cloudwatch_logs(session_id, query=None):
    logs = get_aws_client(session_id, "logs")
    log_groups = []
    paginator = logs.get_paginator("describe_log_groups")

    for page in paginator.paginate():
        for group in page.get("logGroups", []):
            name = group.get("logGroupName")
            if name:
                log_groups.append(name)

    if not query:
        return {
            "log_groups": log_groups[:50],
            "message": "No log group specified."
        }

    query_lower = query.lower()

    matches = [
        group
        for group in log_groups
        if group.lower() in query_lower
        or query_lower in group.lower()
        or any(
            word in group.lower()
            for word in query_lower.split()
            if len(word) > 2
        )
    ]

    if not matches:
        return {
            "error": "No matching CloudWatch log group found.",
            "available_log_groups": log_groups[:50]
        }

    log_group = matches[0]

    response = logs.filter_log_events(
        logGroupName=log_group,
        limit=100
    )

    return {
        "log_group": log_group,
        "events": response.get("events", [])
    }


def get_inspector_findings(session_id, query=None):
    inspector = get_aws_client(session_id, "inspector2")

    response = inspector.list_findings(
        filterCriteria={
            "findingStatus": [
                {
                    "comparison": "EQUALS",
                    "value": "ACTIVE"
                }
            ]
        },
        maxResults=100
    )

    findings = []

    for finding in response.get("findings", []):

        package_vulnerabilities = []

        for vuln in finding.get("packageVulnerabilityDetails", {}).get(
            "vulnerablePackages", []
        ):
            package_vulnerabilities.append({
                "name": vuln.get("name"),
                "version": vuln.get("version"),
                "fixed_version": vuln.get("fixedInVersion"),
                "package_manager": vuln.get("packageManager"),
            })

        resources = []

        for resource in finding.get("resources", []):
            resources.append({
                "type": resource.get("type"),
                "id": resource.get("id"),
                "partition": resource.get("partition"),
                "region": resource.get("region"),
            })

        findings.append({
            "finding_arn": finding.get("findingArn"),
            "title": finding.get("title"),
            "description": finding.get("description"),
            "severity": finding.get("severity"),
            "status": finding.get("status"),
            "type": finding.get("type"),
            "first_observed_at": finding.get("firstObservedAt"),
            "last_observed_at": finding.get("lastObservedAt"),
            "package_vulnerabilities": package_vulnerabilities,
            "resources": resources,
        })

    return {
        "total_findings": len(findings),
        "findings": findings
    }

 