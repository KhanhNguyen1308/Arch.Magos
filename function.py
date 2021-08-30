def count_object(classes):
    face, person = 0, 0
    for object in classes:
        if object[0]== 0:
            face+=1
        if object[0]== 1:
            person+=1
    return face, person
