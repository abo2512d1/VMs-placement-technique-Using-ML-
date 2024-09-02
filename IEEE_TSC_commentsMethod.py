import pandas as pd


HeaderForJobs=['UniqJobId','collection_id','instance_index','priority','machine_id','sample_rate',
         'cpu_usage_distribution1','cpu_usage_distribution2','cpu_usage_distribution3',
         'cpu_usage_distribution4','cpu_usage_distribution5','cpu_usage_distribution6',
         'cpu_usage_distribution7','cpu_usage_distribution8','cpu_usage_distribution9',
         'cpu_usage_distribution10','cpu_usage_distribution11','max_cpu','avg_cpu','cycles_per_instruction',
         'memory_accesses_per_instruction','assigned_memory','page_cache_memory','max_memory','avg_memory',
         'start_time','end_time','time_window','lable']
def proprety_metrices(jodsLables):
    df = pd.DataFrame(jods, columns=HeaderForJobs)
    print("")
