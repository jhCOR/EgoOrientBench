from source.parser import direction_parser

def validation_check(_annotation):
    print("start validation_check")
    delete_list=[]
    num_list = [1,2,3,4,6,7,8,9]
    
    for item in _annotation:
        if(int(item['answer']) not in num_list):
            print("1:", item)
            _annotation.remove(item)
        elif(item['direction'] not in list(direction_parser.keys())):
            print("2:", item)
            _annotation.remove(item)
        elif( (item['direction'] == None) and (item['direction'] == 'None')):
            print("3:", item)
            _annotation.remove(item)

    return _annotation, delete_list