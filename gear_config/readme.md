

### 使用
```
$ 为gear保留字符

修改config/下任意.yaml文件，在.yaml旁自动生成python类文件
from config.YOUR_CONFIG.XXXX.py import ARG
arg = ARG()

from config.YOUR_CONFIG.specificXXXX.py import ARG as sp_ARG
ap_arg = sp_ARG()

from config_utils import merge
arg = merge(arg, sp_arg)
```

### 高级
```
宏功能(可自定义): 
    $time --> time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    $config_name --> "os.path.basename(sys.argv[0])"
    $user --> time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    $project_dir --> dirname(dirname(sys.argv[0]))

指针功能: 
    相对路径: ..b.c
    绝对路径: arg.a.b
    
    exampel: 
    arg:
        a:
            b: 'b_position'
            c: $.b     # -->'b_position'
            d: $..g.x  # -->233
        g:
            x: 233
            
函数功能: 
    python函数均可以在yaml中使用 
    exampel: 
        join(root_path, relative_path)
```

### 修复
```
如果项目文件损坏,不能自动生成python类文件,请按readme_config.png图配置
```
