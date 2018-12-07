
if [ -e "$1" ]; then
 $1/python generate_py_class_file.py
else
 python generate_py_class_file.py
fi
