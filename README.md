GEMM optimization practices
========


### cpu version

following guidelines from https://github.com/flame/how-to-optimize-gemm

```bash
$ cd cpu
$ ./batch_run.sh
$ python ./plot_all.py
```

10~20 times faster than the orignial version(compiled by gcc with `-O2` flag)

![](./opt_result.png)

