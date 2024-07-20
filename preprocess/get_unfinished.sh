python get_all_tars.py $1
aws $CONDUCTOR_ARGS s3 ls $2 | awk '{print $4}' > finished_tar_files.txt
grep -Fxv -f finished_tar_files.txt all_tar_files.txt > unfinished_tar_files.txt