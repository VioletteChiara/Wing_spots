import pickle
from Interface_corrections import find_points

Image_cnt=0
Images=[]


#Folder with images:
Pictures_folder="G:/Pologne/Danny/Butterflies/black_dot_violet/photos"

#TPS file with black dots location:
Black_dots_TPS="G:/Pologne/Danny/Butterflies/black_dot_violet/black_dots.TPS"

#TPS file with Landmarks_location location:
Landmarks_location="G:/Pologne/Danny/Butterflies/black_dot_violet/shape_landmarks31.TPS"

#Where we will save the files
Saving_file="G:\Pologne\Danny\Butterflies\Pickled_data.dan"

with open(Black_dots_TPS) as f:
    lines = f.readlines()
    for line in lines:
        if "LM" in line:#This is a new image
            New_image=[Image_cnt]
            pts=[]
        elif "IMAGE" in line:
            New_image.append(line[6:-1])
        elif "ID" in line:
            New_image.append(int(line[3:-1]))
        elif "SCALE" in line:
            New_image.append(float(line[6:-1]))
            Images.append(New_image+[pts])
            Image_cnt+=1
        else:
            try:
                pos_space=line.index(" ")
                pts.append([float(line[pos_space+1:-1]), float(line[0:pos_space])])
            except:
                if "-1.00" in line:
                    pts.append([-1,-1])

#We now have a table with all coordinates
#We do the same for the wing outer landmarks
Image_cnt=0
pts=[]

with open(Landmarks_location) as f:
    lines = f.readlines()
    for line in lines:
        if not "LM" in line and not "IMAGE" in line and not "ID" in line and not "SCALE" in line:
            try:
                pos_space = line.index(" ")
                pts.append([float(line[pos_space + 1:-1]), float(line[0:pos_space])])
            except:
                if "-1.00" in line:
                    pts.append([-1,-1])
        elif "SCALE" in line:
            Images[Image_cnt][4]=Images[Image_cnt][4]+pts
            Image_cnt+=1
            pts=[]


#Image by image:
imID=0
Show_every=0
for Im in Images:
    Angle,final_cnts,new_cnts, cnt_ID, Fusion=find_points(Im,imID, Pictures_folder ,Show_every)
    Im.append(Angle)  # Save kept contours
    Im.append(final_cnts)#Save kept contours
    Im.append(new_cnts)#Save all contours
    Im.append(cnt_ID)#List of found points
    Im.append(Fusion)  # List of found points
    print(imID)
    imID+=1


with open(Saving_file, 'wb') as fp:
    pickle.dump(Images, fp)
