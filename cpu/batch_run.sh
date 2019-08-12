for f in `ls gemm_*.c`; do
    x=`echo $f | sed 's/.c$//g'`;
    if [ -e "output_$x.m" ] ; then
        echo "$x has already run, skip";
    else
        echo 
        echo "=====> running $x";
        sed -i.bak "s/NEW  := .*/NEW  := $x/g" makefile;
        make run;
    fi
done
