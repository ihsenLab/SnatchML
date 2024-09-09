
#-------------------------------- age - race --------------------------------------#

# echo "---------------- age race simple"
# python3 reprogram_utk.py --original-task age --hijack-task race --model simple ;

# echo "---------------- age race resnet"
# python3 reprogram_utk.py --original-task age --hijack-task race --model resnet ;

echo "---------------- age race mobilenet"
python3 reprogram_utk.py --original-task age --hijack-task race --model mobilenet ;

#-------------------------------- age - gender --------------------------------------#

# echo "---------------- age gender simple"
# python3 reprogram_utk.py --original-task age --hijack-task gender --model simple ;

# echo "---------------- age gender resnet"
# python3 reprogram_utk.py --original-task age --hijack-task gender --model resnet ;

echo "---------------- age gender mobilenet"
python3 reprogram_utk.py --original-task age --hijack-task gender --model mobilenet ;

#-------------------------------- race - gender --------------------------------------#

# echo "---------------- race gender simple"
# python3 reprogram_utk.py --original-task race --hijack-task gender --model simple ;

# echo "---------------- race gender resnet"
# python3 reprogram_utk.py --original-task race --hijack-task gender --model resnet ;

echo "---------------- race gender mobilenet"
python3 reprogram_utk.py --original-task race --hijack-task gender --model mobilenet ;
