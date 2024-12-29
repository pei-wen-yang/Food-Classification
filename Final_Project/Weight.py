import importlib.util
from pathlib import Path
import time
import sys
import RPi.GPIO as GPIO
# from hx711 import HX711

# assign class path
file_path = Path("hx711py/hx711.py")

# dynamic load class
spec = importlib.util.spec_from_file_location("HX711",file_path)
module = importlib.util.module_from_spec(spec)
sys.modules["HX711"] = module
spec.loader.exec_module(module)

# use class
HX711 = module.HX711




def initWeight():
    hx = HX711(5, 6)

    hx.set_reading_format("MSB", "MSB")


    referenceUnit = 427
    hx.set_reference_unit(referenceUnit)

    hx.reset()

    hx.tare()
    
    return hx




def cleanAndExit():
    print("Cleaning...")
        
    print("Bye!")
    sys.exit()
    
def GetWeight(hx):



    print("Tare done! Add weight now...")


    val = hx.get_weight(5)
    print(val)

       
    hx.power_down()
    hx.power_up()
    return val



