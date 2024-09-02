def MigrationCost(memory_size_mb, bandwidth_gbps):
    # Convert memory size from MB to Gb
    memory_size_mb_to_gb = memory_size_mb / 1024  # Convert MB to GB
    memory_size_gb_to_gb = memory_size_mb_to_gb * 8  # Convert GB to Gb

    # Calculate transfer time in seconds
    transfer_time_seconds = memory_size_gb_to_gb / bandwidth_gbps

    # Cost per second is assumed to be 1 unit
    cost_per_second = 1

    # Calculate downtime cost
    downtime_cost = transfer_time_seconds * cost_per_second

    return downtime_cost

# Example usage
memory_size_mb = 0.9  # 0.9 GB in MB
bandwidth_gbps = 5
cost = MigrationCost(memory_size_mb, bandwidth_gbps)
print(f"Downtime Cost: {cost} units")