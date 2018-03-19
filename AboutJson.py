
import json
import inspect 
import pdb

def object2dict(obj):    
    d = {'__class__':obj.__class__.__name__, '__module__':obj.__module__}   
    d.update(obj.__dict__)   
    return d

def objectDumps2File(obj, jsonfile):
    objDict = object2dict(obj)
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(objDict))
    
def dict2object(d):     
    if'__class__' in d:   
        class_name = d.pop('__class__')   
        module_name = d.pop('__module__')   
        module = __import__(module_name)      
        class_ = getattr(module,class_name)   
        args = dict((key.encode('ascii'), value) for key, value in d.items())
        inst = class_(**args)   
    else:   
        inst = d   
    return inst

def objectLoadFromFile(jsonFile):
    with open(jsonFile) as f:
        objectDict =json.load(f)
    obj = dict2object(objectDict)
    return obj

'''  
if __name__  == '__main__':

    class Person(object):   
        def __init__(self,name,age, **args):
            obj_list = inspect.stack()[1][-2]
            self.__name__ = obj_list[0].split('=')[0].strip()#object instance name
            self.name = name   
            self.age = age
            
        def __repr__(self):   
            return 'Person Object name : %s , age : %d' % (self.name,self.age)

        def say(self):
            #d = inspect.stack()[1][-2]
            #print d[0].split('.')[0].strip()
            return self.__name__

        def jsonDumps(self, filename=None):

            if not filename:
                jsonfile = self.__name__+'.json'
            else: jsonfile = filename
            objectDumps2File(self, jsonfile)
        
        def jsonLoadTransfer(self):
            pass


    p = Person('Aidan',22)     
    #json.dumps(p)#error will be throwed
    
    #objectDumps2File(p,'Person.json')
    p.jsonDumps()
    p_l = objectLoadFromFile('p.json')
      
    print 'the decoded obj type: %s, obj:%s' % (type(p_l),repr(p_l))
    '''
