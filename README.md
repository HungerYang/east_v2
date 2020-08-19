## EAST_v2 (revised version)
This code is revised from EAST, with some modifications, including data augmentation, ohem and so on. In post processing, a modified processing different from LANMS is implemented, 
with better result and shorted post-processing time. 

### Major Modifications:
- data enhancement, including horizontal flip, random rotation, mutli-scale, and transpose. 
- hard negative pixel minining
- much quicker geo map genneration, 
- quicker post genenration of quadaraticals
- post processing algorithms.
 
More details can be found in codes.

### How to use this code 
Almost all params are configed in config.py, modify the corresponding params by your own setting.

 **Training:**
 ```python multigpu_trian.py```

**Testing:** 
````python test.py````
 
### More infomation


