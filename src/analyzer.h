
#include <algorithm>
#include <mutex>

extern std::vector<mem_obj_t> g_umem_obj_map;
extern std::vector<mem_obj_t> g_dmem_obj_map;
extern std::map<int, std::vector<mem_obj_t>> g_da_kernel_map;
extern std::map<int, std::vector<mem_obj_t>> g_da_global_map;
extern std::mutex g_mutex;

bool sort_um(mem_obj_t const& lhs, mem_obj_t const& rhs) 
{

  bool res = (lhs.st == rhs.st) ?(lhs.end < rhs.end) : (lhs.st < rhs.st);
  return res;
}

inline void merge_ranges(std::vector<mem_obj_t> &l)
{
  if(l.size()==0) return;
  
  std::sort(l.begin(), l.end(), &sort_um);

  std::vector<mem_obj_t> merged_ranges;
  auto it = l.begin();
  uint64_t merged_range_st = it->st;
  uint64_t merged_range_end= it->end;
  it++;

  for( ; it < l.end() ; ++it){
    //printf("[0x%016lx 0x%016lx] ", sorted_it->st, sorted_it->end);                                                                                                 
    if( it->st <= merged_range_end ){
      merged_range_end = (it->end > merged_range_end) ?(it->end) : merged_range_end;
    }else{
      
      if( merged_range_st != 0x0 )
	merged_ranges.push_back(mem_obj_t({merged_range_st, merged_range_end}));

      merged_range_st = it->st;
      merged_range_end = it->end;
    }
  }
  if( merged_range_st != 0x0 )
    merged_ranges.push_back(mem_obj_t({merged_range_st, merged_range_end}));  

 l = merged_ranges;
}

inline void process_ma_um(mem_access_t *ma, int num_ma){
  
  //printf("process_ma_um %d enter \n", num_ma);
  std::map<int, std::vector<mem_obj_t>> instr_um_map;

  for(int i=0; i<num_ma; i++){
    int memop_size = ma[i].memop_size;
    if(memop_size==-1) break;

    //printf("Instr%d :: memop_size = %d :: ", instr_id, memop_size);
    std::vector<mem_obj_t> l;
    for (int j = 0; j < 32; j++) {
      /* Filter out NULL address */
      if( ma[i].addrs[j]!=0x0 ){
        //printf("0x%016lx ", ma[i].addrs[j]);  
        l.push_back(mem_obj_t({ma[i].addrs[j], ma[i].addrs[j] + memop_size})); 
      }
    }
    //printf("\n");
    merge_ranges(l);

    int instr_id   = ma[i].instr_id;
    auto &it = instr_um_map[instr_id];
    it.insert(it.end(),l.begin(),l.end());
    //printf("Instr%d :: %zu\n", instr_id, it.size());
  }
  
  for(auto &it : instr_um_map){
    //printf("merge[instr%d] length = %zu \n", it.first, (it.second).size());
    merge_ranges(it.second);
  }

  std::lock_guard<std::mutex> guard(g_mutex);
  for(auto it : instr_um_map){
    //printf("merge[instr%d] length = %zu \n", it.first, (it.second).size());
    auto &it2 = g_da_kernel_map[it.first];
    it2.insert(it2.end(), it.second.begin(), it.second.end());
  }
  
  //printf("process_ma_um exit \n");
}

inline void process_ma_cl128(mem_access_t *ma){
  
  std::map<uint64_t,int> hmap;
  for (int i = 0; i < 32; i++) {
    //printf("0x%016lx ", ma->addrs[i]);
    uint64_t a = ma->addrs[i];
    hmap[a] = hmap[a] + 1;
  }

  uint64_t last = 0;
  int excessive_byte=0;
  
  for(auto it : hmap){
    
    int delta = 0;
    if( it.first != 0 && last!=0 ){

      //in the same cache line
      if(it.first/128 == last/128 ){
	delta = (it.first - (last + ma->memop_size));
      }else{
	// in a new cache line
	delta = (128 - (last % 128) - ma->memop_size);
	delta += (it.first % 128);
      }
    }else if(it.first != 0){
      delta = (it.first % 128);
    }
    
    printf("[0x%016lx %d %d] ", it.first, it.second, delta);
    
    last = it.first;
    excessive_byte += delta;
  }
  
  //the last cache line
  excessive_byte += (128 - (last % 128) - ma->memop_size);
  //printf(" excessive_bytes %d \n", excessive_byte);
  printf(" %d ", excessive_byte);
  
}

void process_kernel_data_access() 
{
  std::vector<mem_obj_t> kernel_ranges;

	/* Sort the address ranges for each instruction */
	for( auto it = g_da_kernel_map.begin(); it!=g_da_kernel_map.end(); ++it) {

	  //printf("Instr%d :: %zu \n", it->first, (it->second).size());
	  std::vector<mem_obj_t> &l = it->second;
	  merge_ranges(l);
	  kernel_ranges.insert(kernel_ranges.end(), l.begin(), l.end());

    /* consolidate across all kernel invocations */
    auto &it2 = g_da_global_map[it->first];
    it2.insert(it2.end(), (it->second).begin(), (it->second).end());

	}//End of Instr_map
	    
	merge_ranges(kernel_ranges);
	size_t kernel_used_bytes = 0;
	for(auto r : kernel_ranges)
	  kernel_used_bytes += (r.end - r.st);
	printf("Kernel_used_bytes = %zu\n", kernel_used_bytes);

	int obj_id = 0;
	for(auto obj : g_umem_obj_map) {
	  uint64_t obj_st = obj.st;
	  uint64_t obj_end = obj.end;
	  size_t used_bytes = 0;
	      
	   //printf("UM_OBJ%d [0x%016lx 0x%016lx] = %zu\n", obj_id,obj_st, obj_end, bytes);
	      
	  for(auto it=kernel_ranges.begin(); it<kernel_ranges.end(); it++){
      
      //printf(" [0x%016lx 0x%016lx] \n", it->st, it->end);
		  if( obj_st<=it->st && it->st<obj_end ){        
        uint64_t end = (obj_end<it->end) ?obj_end : it->end; 
		    used_bytes += (end - it->st);
		    it->st = end;
		  }
      
      if( obj_end <= it->st )
		    break;
	  }

	  if( used_bytes > 0 ){
		  //printf("%25s :: used %10zu bytes, unused %10zu bytes\n", 
      //        um_obj_str[obj_id], used_bytes, (obj_end - obj_st - used_bytes));
      printf("\tUM_OBJ%5d :: used %10zu bytes, unused %10zu bytes\n", 
              obj_id, used_bytes, (obj_end - obj_st - used_bytes));
    }
	      
	  obj_id ++;
	    
  } //end of all um_obj

  obj_id = 0;
	for(auto obj : g_dmem_obj_map) {
	  uint64_t obj_st   = obj.st;
	  uint64_t obj_end  = obj.end;
	  size_t used_bytes = 0;
	      
	   //printf("UM_OBJ%d [0x%016lx 0x%016lx] = %zu\n", obj_id,obj_st, obj_end, bytes);
	      
	  for(auto it=kernel_ranges.begin(); it<kernel_ranges.end(); it++){
      
      //printf(" [0x%016lx 0x%016lx] \n", it->st, it->end);
		  if( obj_st<=it->st && it->st<obj_end ){        
        uint64_t end = (obj_end<it->end) ?obj_end : it->end; 
		    used_bytes += (end - it->st);
		    it->st = end;
		  }
      
      if( obj_end <= it->st )
		    break;
	  }

	  if( used_bytes > 0 ){
      printf("\tD_OBJ%5d :: used %10zu bytes, unused %10zu bytes\n", 
              obj_id, used_bytes, (obj_end - obj_st - used_bytes));
    }
	      
	  obj_id ++;
	    
  } //end of all device mem obj
	    
	/*Clear records for this kernel invocation*/
	g_da_kernel_map.clear();
}



void process_global_data_access() 
{

  printf("\n\nEnter process_global_data_access\n");
  std::vector<mem_obj_t> global_ranges;

	/* Sort the address ranges for each instruction */
	for( auto it = g_da_global_map.begin(); it!=g_da_global_map.end(); ++it) {

	  std::vector<mem_obj_t> &l = it->second;
	  merge_ranges(l);
	  global_ranges.insert(global_ranges.end(), l.begin(), l.end());

	}//End of Instr_map
	    
	merge_ranges(global_ranges);

	size_t used_bytes = 0;
	for(auto r : global_ranges)
	  used_bytes += (r.end - r.st);
	printf("Total used_bytes = %zu\n", used_bytes);

	int obj_id = 0;
	for(auto obj : g_umem_obj_map) {
	  uint64_t obj_st = obj.st;
	  uint64_t obj_end = obj.end;
	  size_t used_bytes = 0;
	      
	   //printf("UM_OBJ%d [0x%016lx 0x%016lx] = %zu\n", obj_id,obj_st, obj_end, bytes);
	      
	  for(auto it=global_ranges.begin(); it<global_ranges.end(); it++){
      
      //printf(" [0x%016lx 0x%016lx] \n", it->st, it->end);
		  if( obj_st<=it->st && it->st<obj_end ){        
        uint64_t end = (obj_end<it->end) ?obj_end : it->end; 
		    used_bytes += (end - it->st);
		    it->st = end;
		  }
      
      if( obj_end <= it->st )
		    break;
	  }

	  if( used_bytes > 0 ){
		  //printf("%25s :: used %10zu bytes, unused %10zu bytes\n", 
      //        um_obj_str[obj_id], used_bytes, (obj_end - obj_st - used_bytes));
      printf("\tUM_OBJ%5d :: used %10zu bytes, unused %10zu bytes\n", 
              obj_id, used_bytes, (obj_end - obj_st - used_bytes));
    }
	      
	  obj_id ++;
	    
  } //end of all um_obj

  obj_id = 0;
	for(auto obj : g_dmem_obj_map) {
	  uint64_t obj_st = obj.st;
	  uint64_t obj_end = obj.end;
	  size_t used_bytes = 0;
	      
	   //printf("UM_OBJ%d [0x%016lx 0x%016lx] = %zu\n", obj_id,obj_st, obj_end, bytes);
	      
	  for(auto it=global_ranges.begin(); it<global_ranges.end(); it++){
      
      //printf(" [0x%016lx 0x%016lx] \n", it->st, it->end);
		  if( obj_st<=it->st && it->st<obj_end ){        
        uint64_t end = (obj_end<it->end) ?obj_end : it->end; 
		    used_bytes += (end - it->st);
		    it->st = end;
		  }
      
      if( obj_end <= it->st )
		    break;
	  }

	  if( used_bytes > 0 ){
      printf("\tD_OBJ%5d :: used %10zu bytes, unused %10zu bytes\n", 
              obj_id, used_bytes, (obj_end - obj_st - used_bytes));
    }
	      
	  obj_id ++;
	    
  } //end of all device mem obj
	    
	/*Clear records */
	g_da_global_map.clear();
}


int find_data_obj( uint64_t addr ){
  int obj_id = 0;
	for(auto obj : g_dmem_obj_map) {
	  uint64_t obj_st = obj.st;
	  uint64_t obj_end = obj.end;
      
    if( obj_st <= addr && addr < obj_end )
		  break;
	      
	  obj_id ++;
	    
  } //end of all device mem obj

  if( obj_id >= (int) g_dmem_obj_map.size()){
    //printf("Error :: data movement from unknow Device Memory Object %lx\n", addr);
    obj_id = -1;
  }

  return obj_id;
}