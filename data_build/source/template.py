def getSimpleTemplate(_context):
    _prompt = '''\n
    As a capable vision-language model assistant, your task is to closely examine key features of the object in the provided image and perform the following actions: 1) identify and ask questions about the features that indicate the object’s orientation, and 2) answer the question yourself.

    To elaborate, you should carefully examine the details that suggest the front or rear of the object, such as eyes, nose, mouth, or tail, or tail lights. Additionally, you should closely check for features that imply a orientation, such as the orientation of the nose, whether one or both arms of a person are visible, or whether the wheels of a car appear as perfect circles, indicating left or right.

    Instead of making overly general statements, you must create responses that are detailed enough to determine a single orientation. The answer should not just state that something is visible, but rather explain how these features suggest a particular orientation and provide specific details to justify the object's orientation.

    For example, you could write something like this:
    [Question]: Describe in detail the features in the image that indicate the orientation of the object.
    [Answer]: The two wheels of the car appear perfectly circular, and the car door is visible. This suggests that the object's orientation is either "left" or "right." However, since the headlights are on the right side of the image and the red taillights are on the left, the car's orientation is definitively to the "right."

    [Question]: Describe in detail the features in the image that indicate the orientation of the object.
    [Answer]: Both eyes of the person’s face are visible, but one eye appears larger, and only one cheek is primarily visible, indicating a slanted orientation. The pointed part of the nose is closer to the left side of the image, and the left cheek is not clearly visible from the camera’s perspective. Therefore, the person is facing "front left."

    Now your Turn: 
    ''' + _context + "\n[Question]: "+"###\n[Answer]: "
    return _prompt

def getComplexResoningTemplate(_context):
    _prompt = '''
    \nAs a competent helper like Turn-by-turn navigation, you have the role of understanding the properties of the central object and creating common sense questions and corresponding answer appropriate for them.
    You must use information in [Context] If you make a question and answer, think carefully that matching the object’s property and common sense with the probable action or capable happening.
    But pay attention that while making question, you MUST not contain the orientation information directly you get from [Context] in question.

    Alternatively you can represent with indirect word like corner of image, behind of it, away from camera’s view point or something else.
    Also, use natural expressions for orientation expressions like facing away from the camera, facing right else.
    You must now generate a question, similar to the given examples, asking which orientation the object should be turned to face or turn away from the camera with the least angle of rotation, along with the corresponding answer.
   
    I’ll give you some example that you can reference.
    For example, If you find the main object as a sport car, you can make a question like this:
    
    [Question]: To make an object face the camera directly with the smallest rotation angle, in which direction should
    it turn? Choose from [clockwise, counterclockwise, flip, leave as is] and explain why. 
    [Answer]: The sport car is facing to the back, so you have to flip it.
    
    [Question]: To make an object face the camera directly with the smallest rotation angle, in which direction should it turn? Choose from [clockwise, counterclockwise, flip, leave as is] and explain why.
    [Answer]: The sport car is
    facing to the right, so you have to turn to the counterclockwise to look straight at the camera.

    
    Now your Turn: 
    '''  + _context + "\n[Question]: "+"###\n[Answer]: "
    return _prompt

def getDetailTemplate(_context):
    _prompt = '''\nAs a competent assistant, your role is to explain the subparts of the main object and, based on your findings, determine its orientation.
    
    These parts of the object should eventually be able to imply some orientation of the object, but questions should never directly include information about its orientation.
    You must use the information in [Context], but the important thing is that you must find subparts or sub-features of the object given in [Context].
    If you identify the main object as a human, you need to find different subparts depending on the orientation the person is facing in the picture. 
    For example, if the person is facing forward (i.e., looking at the camera), you will see the face, eyes, chest, or abdomen. 
    If the person is facing backward, you will see the hip, back, or hair. If the person is facing to the right, 
    you will see one ear and one arm, and you must note that the nose is pointing to the right.

    For example,
    [Question]: From camera perspective, does the {sport car} is {facing camera} or {facing away} the camera/observer? First describe what you can find from the object in the image, then based on that, answer the orientation of the object. 
    [Answer]: 
    (What I find): headlight, windshield, bumper
    (Answer the orientation): According to (What I find), The {sport car} facing the camera/observe.
    
    [Question]: Does the {a girl} is facing left or facing right from camera perspective? First describe what you can find from the object in the image, then based on that, answer the orientation of the object. 
    [Answer]: 
    (finding): hair, hip, arm, half of nose
    (Answer the orientation): According to (What I find), The {a girl} facing the left.
    
    Now your turn:
    ''' +  _context + "\n[Question]: " + "###\n[Answer]:  "
    return _prompt

def getTemplate(index, context):
    if(index == 0):
        return getSimpleTemplate(context)
    elif(index == 1):
        return getComplexResoningTemplate(context)
    elif(index == 2):
        return getDetailTemplate(context)
    elif(index == -1):
        return {
            "Simple": getSimpleTemplate(context),
            "ComplexResoning": getComplexResoningTemplate(context),
            "Detail": getDetailTemplate(context)
            }