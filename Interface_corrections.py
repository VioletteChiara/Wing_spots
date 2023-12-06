from tkinter import *
import cv2
import math
import PIL
import numpy as np
import pickle
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
from functools import partial
from tkinter import filedialog as fd, messagebox
import csv
from sklearn.cluster import KMeans
# create root window
import scipy as sc

Normalised_scale = 0.04




def find_points(Im, imID, Path, Show_every=0, Angle="NA"):
    File=Im[1]
    img=cv2.imread(Path +"/"+ File)
    change_to_do = Normalised_scale / Im[3]
    img = cv2.resize(img, (int(img.shape[1] / change_to_do), int(img.shape[0] / change_to_do)))


    known_coos_all = [[pt[0]/change_to_do,pt[1]/change_to_do] for pt in Im[4] if pt[0] != -1]
    known_coos_spots = [[pt[0]/change_to_do,pt[1]/change_to_do] for pt in Im[4][0:9] if pt[0] != -1]
    unknown_coos_spots=[pt for pt in range(len(Im[4][0:9])) if Im[4][pt][0] == -1]
    ID_coos_spots=[pt for pt in range(len(Im[4][0:9])) if Im[4][pt][0] != -1]


    empty1=np.zeros((img.shape[0],img.shape[1],1), np.uint8)
    new_list=[[[int(pt[0]), int(pt[1])]] for pt in known_coos_all if pt[0]!=-1]
    new_list=np.array(new_list, dtype="int32")
    hull=cv2.convexHull(new_list, False)


    empty2=cv2.rotate(empty1, cv2.ROTATE_90_CLOCKWISE)
    empty3 = cv2.rotate(empty2, cv2.ROTATE_90_CLOCKWISE)
    empty4 = cv2.rotate(empty3, cv2.ROTATE_90_CLOCKWISE)

    empty1 = cv2.drawContours(empty1,[hull],-1,255,-1)
    empty2 = cv2.drawContours(empty2, [hull], -1, 255, -1)
    empty3 = cv2.drawContours(empty3,[hull],-1,255,-1)
    empty4 = cv2.drawContours(empty4, [hull], -1, 255, -1)
    masks = [empty1, empty2, empty3, empty4]

    grey1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    grey2=cv2.rotate(grey1, cv2.ROTATE_90_CLOCKWISE)
    grey3=cv2.rotate(grey2, cv2.ROTATE_90_CLOCKWISE)
    grey4=cv2.rotate(grey3, cv2.ROTATE_90_CLOCKWISE)

    maskT1 = empty1[:, :, 0].astype(bool)
    maskT2 = empty2.astype(bool)
    maskT3 = empty3.astype(bool)
    maskT4 = empty4.astype(bool)
    binmasks = [maskT1, maskT2, maskT3, maskT4]

    size=cv2.contourArea(hull)

    Prop_Whites1=sum(grey1[maskT1,0]>120)/size
    Prop_Whites2 = sum(grey2[maskT2, 0]>120) / size
    Prop_Whites3 = sum(grey3[maskT3, 0]>120) / size
    Prop_Whites4 = sum(grey4[maskT4, 0]>120) / size

    Liste_prop=[Prop_Whites1,Prop_Whites2,Prop_Whites3,Prop_Whites4]
    Mini=min(Liste_prop)
    if Angle == "NA":
        Angle=Liste_prop.index(Mini)


    for i in range(Angle):
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    or_img=np.copy(img)



    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (7,7))
    img=cv2.bitwise_and(img,img,mask=masks[Angle])

    bright=(np.sum(img[binmasks[Angle]]) / (255 * img[binmasks[Angle]].size))
    ratio = bright / 0.30
    img = cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)

    img=cv2.adaptiveThreshold(img,  255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5.6)
    cnts,_ =cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_cnts=[]
    for cnt in cnts:
        surface=cv2.contourArea(cnt)*Normalised_scale
        if surface>0 and surface<500:
            new_cnts.append(cnt)

    final_cnts, ID_coos_spots, Fusion=Identify_Pts(Im, new_cnts, known_coos_spots, ID_coos_spots, imID)


    if  Show_every>0 and imID%Show_every==0:
        red_img = np.copy(or_img)
        red_img = cv2.drawContours(red_img, new_cnts, -1, (0, 255, 255), -1)
        red_img=cv2.drawContours(red_img, final_cnts,-1,(0,0,255),-1)
        #red_img = cv2.drawContours(red_img, final_cnts, -1, (0, 0, 0), 1)

        alpha=0.5
        red_img=cv2.addWeighted(or_img, alpha, red_img, 1 - alpha, 0)

        for pt in range(len(ID_coos_spots)):
            M = cv2.moments(final_cnts[pt])
            if M["m00"]==0:M["m00"]=1
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            red_img=cv2.putText(red_img, str(ID_coos_spots[pt]+1), (int(cX+5),int(cY+5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

        cv2.imshow("TEST", cv2.resize(red_img, (int(red_img.shape[1]/1.5), int(red_img.shape[0]/1.5))))
        cv2.waitKey()

    return(Angle, final_cnts, new_cnts, ID_coos_spots, Fusion)


def Identify_Pts(Im, new_cnts, known_coos_spots, ID_coos_spots, imID):
    Pres_8_24= Im[4][7][0]!=-1 and Im[4][23][0]!=-1
    Pres_0_1= Im[4][0][0]!=-1 and Im[4][8][0]!=-1
    Fusion=False

    centers=[]
    areas=[]
    for cnt in new_cnts:
        M = cv2.moments(cnt)
        if M["m00"]==0:M["m00"]=1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
        areas.append(cv2.contourArea(cnt) * Normalised_scale)


    #We know look for the facultative points:
    if Pres_0_1 and 9 not in ID_coos_spots:
        look_around= (int((2*Im[4][0][0]/3 + Im[4][8][0]/3)/(Normalised_scale / Im[3])), int((2*Im[4][0][1]/3 + Im[4][8][1]/3)/(Normalised_scale / Im[3])))
        known_coos_spots.append(look_around)
        ID_coos_spots.append(9)

    if Pres_8_24 and 10 not in ID_coos_spots:
        look_for = int((4 * Im[4][7][0] / 5 + Im[4][24][0] / 5)/(Normalised_scale / Im[3])), int((4 * Im[4][7][1] / 5 + Im[4][24][1] / 5)/(Normalised_scale / Im[3]))
        known_coos_spots.append(look_for)
        ID_coos_spots.append(10)


    if len(centers)>0:
        dists = sc.spatial.distance_matrix(known_coos_spots,centers)
        row_ind, col_ind = sc.optimize.linear_sum_assignment(dists)
        if len(row_ind)<len(ID_coos_spots):
            ID_coos_spots=[ID_coos_spots[i] for i in row_ind]
        final_cnts = [new_cnts[i] for i in col_ind]
        final_areas = [areas[i] for i in col_ind]

        to_pop=[]
        for ind in range(len(row_ind)):
            if dists[row_ind[ind]][col_ind[ind]]>0.4/Normalised_scale:
                to_pop.append(ind)

        if len(to_pop)>0:
            to_pop.reverse()
            for ind in to_pop:
                row_ind=np.delete(row_ind,ind)
                col_ind=np.delete(col_ind,ind)
                final_areas=np.delete(final_areas, ind)
                if len(final_cnts)>1:
                    final_cnts=np.delete(final_cnts, ind)
                else:
                    final_cnts=[]
                ID_coos_spots=np.delete(ID_coos_spots,ind)

        # Calculate the circularity:
        final_circularity = []
        for idcnt in range(len(final_cnts)):
            area = final_areas[idcnt] / Normalised_scale
            arclength = cv2.arcLength(final_cnts[idcnt], True)
            if arclength * arclength >0:
                circularity = (4 * math.pi * area) / (arclength * arclength)
            else:
                circularity=0
            final_circularity.append(circularity)

        try:
            To_sep = list(ID_coos_spots).index(7)
        except:
            To_sep = -1
        final_cnts = list(final_cnts)
        ID_coos_spots = list(ID_coos_spots)

        if (To_sep) >= 0:
            if 10 not in ID_coos_spots and final_circularity[To_sep] < 0.70:
                array = np.vstack([final_cnts[To_sep]])
                array = array.reshape(array.shape[0], array.shape[2])
                kmeans = KMeans(n_clusters=2, random_state=1000, n_init=20).fit(array)
                cnts_class = kmeans.fit_predict(array)
                cnts_center = kmeans.cluster_centers_
                cnts_s = [array[np.where(cnts_class == 0)], array[np.where(cnts_class == 1)]]

                dists = sc.spatial.distance_matrix([look_for, Im[4][7]], cnts_center)
                row_ind2, col_ind2 = sc.optimize.linear_sum_assignment(dists)

                final_cnts[To_sep] = cnts_s[col_ind2[1]]
                final_cnts.append(cnts_s[col_ind2[0]])
                row_ind = np.append(row_ind, len(final_cnts) - 1)
                ID_coos_spots.append(10)
                Fusion = True

    else:
        print("ERROR, no points found:" + str(imID))
        final_cnts=[]
        ID_coos_spots=[]
        Fusion=0


    return (final_cnts, ID_coos_spots, Fusion)



class Details_inter(Frame):
    def __init__(self, parent, **kwargs):
        Frame.__init__(self, parent, bd=5, **kwargs)
        self.parent=parent
        self.grid(sticky="nsew")
        Grid.columnconfigure(self.parent, 0, weight=1)  ########NEW
        Grid.rowconfigure(self.parent, 0, weight=1)  ########NEW
        self.ready=False
        self.parent.attributes('-toolwindow', True)

        self.final_width = 250
        self.zoom_strength = 0.3

        #organization of the Frame
        self.Canvas_for_video = Canvas(self, width=1500, height=800, bd=0, highlightthickness=0)
        self.Canvas_for_video.grid(row=0, column=0, sticky="nsew")
        Grid.columnconfigure(self, 0, weight=1)  ########NEW
        Grid.columnconfigure(self, 1, weight=1)  ########NEW
        Grid.rowconfigure(self, 1, weight=1)  ########NEW
        self.Canvas_for_video.update()

        self.Canvas_for_video.bind("<Button-1>", self.callback)
        self.Canvas_for_video.bind("<Button-3>", self.Rcallback)

        self.Canvas_for_video.bind("<Motion>", self.Move)
        self.Canvas_for_video.bind("<B1-Motion>", self.Move_N_Draw)
        self.Canvas_for_video.bind("<B3-Motion>", self.Move_N_Erase)
        self.Canvas_for_video.bind("<Control-1>", self.Zoom_in)
        self.Canvas_for_video.bind("<Control-3>", self.Zoom_out)
        self.Canvas_for_video.bind("<Configure>", self.show_img)

        self.Canvas_for_video.bind("<MouseWheel>", self.On_mousewheel)

        self.bind_all("<space>", self.hide_cnts)

        self.Frame_user = Frame(self, width=150)
        self.Frame_user.grid(row=0, column=1, rowspan=2, sticky="nsew")
        Grid.columnconfigure(self.Frame_user, 0, weight=1)  ########NEW
        Grid.rowconfigure(self.Frame_user, 0, weight=1)  ########NEW
        Grid.rowconfigure(self.Frame_user, 1, weight=1)  ########NEW
        Grid.rowconfigure(self.Frame_user, 2, weight=100)  ########NEW


        # Help user and parameters
        Frame_Ana = Frame(self.Frame_user)
        Frame_Ana.grid(row=2, column=0, columnspan=2, sticky="nsew")
        Grid.columnconfigure(Frame_Ana, 0, weight=1)  ########NEW
        Grid.rowconfigure(Frame_Ana, 0, weight=1)  ########NEW


        Fr_Right=Frame(self)
        Fr_Right.grid(row=0,column=1,sticky="nsew")
        Grid.columnconfigure(self, 0, weight=1)  ########NEW
        Grid.rowconfigure(self, 0, weight=1)  ########NEW

        self.alpha_val=IntVar()
        Transp_scale=Scale(Fr_Right,label="Transparency",from_=1, to=100, variable=self.alpha_val, orient=HORIZONTAL, command=self.show_img)
        Transp_scale.grid()


        self.show_cnt=True

        self.ready = True
        self.wait_click=-1
        self.is_drawing=False
        self.tool_size=20


        self.Saved_File=fd.askopenfilename()
        with open(self.Saved_File, 'rb') as fp:
                self.Images = pickle.load(fp)


        self.NB_Frame=sum([1 for num in range(len(self.Images)) if len(self.Images[num])>8])

        self.Path_Images = fd.askdirectory()

        Frame_change_img=Frame(Fr_Right)
        Frame_change_img.grid()
        self.En_Pos=Entry(Frame_change_img, text="0")
        self.En_Pos.grid(row=0, column=1, sticky="nsew")
        self.cur_num=0
        Button(Frame_change_img,text="Change image", command=self.change_Img).grid(row=1,column=1)

        Next_Button=Button(Frame_change_img, text="->", command=self.next)
        Next_Button.grid(row=0, column=2)

        Prev_Button=Button(Frame_change_img, text="<-", command=self.prev)
        Prev_Button.grid(row=0, column=0)



        Frame_button=Frame(Fr_Right)
        Frame_button.grid(row=3,column=0, sticky="nsew")
        self.Pres_Buttons=[]
        for Pt in range(11):
            if Pt<9:
                Label(Frame_button,text="Point: "+str(Pt+1)).grid(row=Pt,column=0)
            elif Pt==9:
                Label(Frame_button, text="Facultative 1: " + str(Pt + 1)).grid(row=Pt, column=0)
            elif Pt==10:
                Label(Frame_button, text="Facultative 2: " + str(Pt + 1)).grid(row=Pt, column=0)
                self.Fus_but=Button(Frame_button, text="Fusionned", background="green", command=self.change_Fus, width=15)
                self.Fus_but.grid(row=Pt, column=2)

            self.Pres_Buttons.append(Button(Frame_button,command=partial(self.change_status,Pt), width=15))
            self.Pres_Buttons[Pt].grid(row=Pt,column=1)

        Draw_Button=Button(Fr_Right, text="Drawing", command=self.drawing, background="yellow")
        Draw_Button.grid()

        Redo_Button=Button(Fr_Right, text="Redo original", command=self.redo)
        Redo_Button.grid()

        Save_N_next_Button=Button(Fr_Right, text="Save and continue", command=self.contin)
        Save_N_next_Button.grid()


        self.Corrected=Button(Fr_Right,text="Not corrected", fg="red", command=self.remove_correction)
        self.Corrected.grid()

        RotateB=Button(Fr_Right,text="↷", command=self.rotate)
        RotateB.grid()

        self.load_img(self.cur_num)
        self.show_img()
        self.update_buttons()

        Button(Fr_Right, text="Save as csv", command=self.save_tables).grid()

    def rotate(self):
        self.Images[self.cur_num][5]+=1
        if self.Images[self.cur_num][5]>=4:
            self.Images[self.cur_num][5]-=4
        self.load_img(self.cur_num)
        self.redo()


    def remove_correction(self):
        self.Images[self.cur_num][10]=0
        with open(self.Saved_File, 'wb') as fp:
            pickle.dump(self.Images, fp)
        self.update_buttons()

    def save_tables(self):
        To_save= fd.asksaveasfilename(defaultextension=".TPS", initialfile="Untitled_tps.TPS", filetypes=(("TPS", "*.TPS"),))
        with open(To_save, 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            for Im in self.Images:
                if len(Im)>9:
                    change_to_do = 0.04 / Im[3]
                    writer.writerow(["LM=11"])
                    for Pt in range(11):
                        if Pt in Im[8]:
                            cnt=Im[6][Im[8].index(Pt)]
                            M = cv2.moments(cnt)
                            if M["m00"] == 0: M["m00"] = 1
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cX= int(cX*change_to_do)
                            cY = int(cY * change_to_do)
                            writer.writerow([f'{cY:.5f}' + " " + f'{cX:.5f}'])
                        else:
                            writer.writerow(["-1.00000 -1.00000"])

                    writer.writerow(["IMAGE="+str(Im[1])])
                    writer.writerow(["ID="+str(Im[2])])
                    writer.writerow(["SCALE=" + f'{Im[3]:.5f}'])

        To_save=To_save[:-4]+"_All_points.TPS"
        with open(To_save, 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            for Im in self.Images:
                if len(Im)>9:
                    change_to_do = 0.04 / Im[3]
                    writer.writerow(["LM=42"])
                    for Pt in Im[4][9:]:
                        writer.writerow([f'{Pt[1]:.5f}' + " " + f'{Pt[0]:.5f}'])

                    for Pt in range(11):
                        if Pt in Im[8]:
                            cnt=Im[6][Im[8].index(Pt)]
                            M = cv2.moments(cnt)
                            if M["m00"] == 0: M["m00"] = 1
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cX= int(cX*change_to_do)
                            cY = int(cY * change_to_do)
                            writer.writerow([f'{cY:.5f}' + " " + f'{cX:.5f}'])
                        else:
                            writer.writerow(["-1.00000 -1.00000"])

                    writer.writerow(["IMAGE="+str(Im[1])])
                    writer.writerow(["ID="+str(Im[2])])
                    writer.writerow(["SCALE=" + f'{Im[3]:.5f}'])

        To_save=To_save[:-3]+"csv"
        with open(To_save, 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["Photo_ID","File_name","Point_ID","Area","Circularity","Dot_type","Presence","Fusion","Corrected"])
            for Im in self.Images:
                if len(Im) > 9:
                    for spot in range(11):
                        new_row = []
                        new_row.append(Im[2])#Photo Id
                        new_row.append(Im[1])#File name
                        new_row.append(spot+1)#Pt name

                        #Area + Circ
                        if spot in Im[8]:
                            Presence=1
                            cnt = Im[6][Im[8].index(spot)]
                            Ar=cv2.contourArea(cnt)
                            new_row.append(Ar * Normalised_scale)#Area
                            arclength = cv2.arcLength(cnt, True)
                            circularity = (4 * math.pi * Ar) / (arclength * arclength)
                            new_row.append(circularity)#Circu
                        else:
                            Presence=0
                            new_row.append("NA")
                            new_row.append("NA")

                        if spot == 9 or spot ==10:
                            new_row.append("facultative")
                        else:
                            new_row.append("normal")

                        new_row.append(Presence)

                        if spot==10 and Im[9]:#Fusion
                            new_row.append(1)
                        else:
                            new_row.append(0)

                        if len(Im)>10 and Im[10]:
                            new_row.append(1)
                        else:
                            new_row.append(0)

                        writer.writerow(new_row)



    def change_Img(self):
        try:
            if int(self.En_Pos.get())<self.NB_Frame:
                self.cur_num = int(self.En_Pos.get())
                self.load_img(int(self.En_Pos.get()))

        except:
            count=0
            for Im in self.Images:
                if Im[1]==self.En_Pos.get() and count<self.NB_Frame:
                    self.cur_num = count
                    self.load_img(count)
                    self.update_buttons()
                    break
                count+=1

    def change_Fus(self):
        self.Fusion=1-self.Fusion
        self.update_buttons()

    def contin(self):
        self.Images[self.cur_num][6]=self.kept_cnts
        self.Images[self.cur_num][7] = self.all_cnts
        self.Images[self.cur_num][8] = self.found_pts
        self.Images[self.cur_num][9] = self.Fusion
        self.Images[self.cur_num][10] = True
        with open(self.Saved_File, 'wb') as fp:
            pickle.dump(self.Images, fp)

        self.next()

    def next(self):
        if self.cur_num<self.NB_Frame-1:
            self.cur_num+=1
            self.load_img(self.cur_num)
        else:
            messagebox.showinfo(title="Finished!", message="Congratulations, you reached the last image!")


    def prev(self):
        if self.cur_num>0:
            self.cur_num-=1
            self.load_img(self.cur_num)


    def redo(self):
        Angle, self.kept_cnts, self.all_cnts, self.found_pts, self.Fusion= find_points(self.Images[self.cur_num],0, self.Path_Images,0, Angle=self.Images[self.cur_num][5])
        self.empty=np.zeros((self.image.shape[0],self.image.shape[1],1), dtype = "uint8")
        self.empty=cv2.drawContours(self.empty,self.all_cnts,-1,255,-1)
        self.show_img()
        self.update_buttons()

    def update_buttons(self):
        for Pt in range(11):
            self.show_B_status(Pt)
        if self.Fusion:
            self.Fus_but.config(background="green", text="There is fusion")
        else:
            self.Fus_but.config(background="red", text="No fusion")

        if self.Images[self.cur_num][10]:
            self.Corrected.config(fg="green", text="Corrected")
        else:
            self.Corrected.config(fg="red", text="Not Corrected")

        self.En_Pos.delete(0,END)
        self.En_Pos.insert(0, self.cur_num)


    def drawing(self):
        if self.is_drawing:
            change_to_do = 0.04 / self.Images[self.cur_num][3]

            Im=self.Images[self.cur_num]
            cnts, _ = cv2.findContours(self.empty, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            known_coos_spots = [[pt[0]/change_to_do, pt[1]/change_to_do] for pt in Im[4][0:9] if pt[0] != -1]
            ID_coos_spots = [pt for pt in range(len(Im[4][0:9])) if Im[4][pt][0] != -1]
            self.kept_cnts, self.found_pts, self.Fusion=Identify_Pts(Im, cnts, known_coos_spots, ID_coos_spots, 0)
            self.all_cnts=cnts
            self.update_buttons()


        self.is_drawing=1-self.is_drawing
        self.show_img()

    def callback(self, event):
        if self.wait_click!=-1:
            PtX = int(event.widget.canvasx(event.x) * self.ratio + self.zoom_sq[0])
            PtY = int(event.widget.canvasy(event.y) * self.ratio + self.zoom_sq[1])
            for cnt in self.all_cnts:
                isIn=cv2.pointPolygonTest(cnt,[PtX,PtY], measureDist=False)
                if isIn!=-1:
                    self.found_pts.append(self.wait_click)
                    self.kept_cnts.append(cnt)
                    self.show_B_status(self.wait_click)
                    self.show_img()
                    self.wait_click=-1
                    break
        if self.is_drawing:
            PtX = int(event.widget.canvasx(event.x) * self.ratio + self.zoom_sq[0])
            PtY = int(event.widget.canvasy(event.y) * self.ratio + self.zoom_sq[1])
            self.empty=cv2.circle(self.empty,[PtX,PtY],self.tool_size,255,-1)
            self.show_img()

    def Rcallback(self, event):
        if self.is_drawing:
            PtX = int(event.widget.canvasx(event.x) * self.ratio + self.zoom_sq[0])
            PtY = int(event.widget.canvasy(event.y) * self.ratio + self.zoom_sq[1])
            self.empty=cv2.circle(self.empty,[PtX,PtY],self.tool_size,0,-1)
            self.show_img()

    def On_mousewheel(self, event):
        if event.delta>0 or (self.tool_size>0.5 and event.delta<0):
            self.tool_size = int(self.tool_size  + (event.delta / 60))
        self.show_img(self.cur_pos)


    def change_status(self, BID):
        if BID in self.found_pts:
            pos=self.found_pts.index(BID)
            self.kept_cnts.pop(pos)
            self.found_pts.pop(pos)
        elif self.wait_click==BID:
            self.wait_click=-1
        else:
            self.wait_click=BID
        self.show_img()
        self.show_B_status(BID)


    def show_B_status(self,BID):
        if BID in self.found_pts:
            self.Pres_Buttons[BID].config(background="green", text="Present")
        elif BID==self.wait_click:
            self.Pres_Buttons[BID].config(background="grey", text="Waiting")
        else:
            self.Pres_Buttons[BID].config(background="red", text="Absent")

    def hide_cnts(self, _):
        self.show_cnt = 1-self.show_cnt
        self.show_img()

    def load_img(self, num):
        try:
            File = self.Images[num][1]
            if len(self.Images[num])==10:
                self.Images[num].append(False)
            self.image=cv2.imread(self.Path_Images +"/"+ File)
            change_to_do = 0.04 / self.Images[num][3]
            self.image=cv2.resize(self.image,(int(self.image.shape[1]/change_to_do),int(self.image.shape[0]/change_to_do)))

            self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
            self.parent.title("Image number: " + str(num) + "         "+ "File: " + File)
            Angle=self.Images[self.cur_num][5]

            for i in range(Angle):
                self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)

            self.Size = self.image.shape
            self.ratio = self.Size[1] / self.final_width
            self.zoom_sq = [0, 0, self.image.shape[1], self.image.shape[0]]
            self.kept_cnts=self.Images[num][6]
            self.all_cnts = self.Images[num][7]
            self.found_pts = self.Images[num][8]
            self.Fusion = self.Images[num][9]
            self.empty=np.zeros((self.image.shape[0],self.image.shape[1],1), dtype = "uint8")
            self.empty=cv2.drawContours(self.empty,self.all_cnts,-1,255,-1)
            self.show_img()
            self.update_buttons()

        except Exception as e:
            self.next()


    def Move(self,event):
        if self.is_drawing:
            PtX = int(event.widget.canvasx(event.x) * self.ratio + self.zoom_sq[0])
            PtY = int(event.widget.canvasy(event.y) * self.ratio + self.zoom_sq[1])
            self.cur_pos=[PtX,PtY]
            self.show_img([PtX,PtY])

    def Move_N_Draw(self, event):
        if self.is_drawing:
            PtX = int(event.widget.canvasx(event.x) * self.ratio + self.zoom_sq[0])
            PtY = int(event.widget.canvasy(event.y) * self.ratio + self.zoom_sq[1])
            self.cur_pos=[PtX,PtY]
            self.empty = cv2.circle(self.empty, [PtX, PtY], self.tool_size, 255, -1)
            self.show_img([PtX,PtY])

    def Move_N_Erase(self, event):
        if self.is_drawing:
            PtX = int(event.widget.canvasx(event.x) * self.ratio + self.zoom_sq[0])
            PtY = int(event.widget.canvasy(event.y) * self.ratio + self.zoom_sq[1])
            self.cur_pos=[PtX,PtY]
            self.empty = cv2.circle(self.empty, [PtX, PtY], self.tool_size, 0, -1)
            self.show_img([PtX,PtY])

    def show_img(self, cur_pos="NA", *args):
        #Display the image
        try:
            if self.show_cnt:
                if not self.is_drawing:
                    red_img = np.copy(self.image)
                    red_img = cv2.drawContours(red_img, self.all_cnts, -1, (0, 255, 255), -1)
                    red_img = cv2.drawContours(red_img, self.kept_cnts, -1, (0, 0, 255), -1)
                    red_img = cv2.drawContours(red_img, self.kept_cnts, -1, (255,0,0), int(self.ratio*2))
                if self.is_drawing:
                    cnts, _ = cv2.findContours(self.empty, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    red_img = np.copy(self.image)
                    red_img = cv2.drawContours(red_img, cnts, -1, (0,255,255),-1)

                alpha=self.alpha_val.get()/100
                red_img=cv2.addWeighted(self.image, alpha, red_img, 1 - alpha, 0)
                im_to_show=red_img

                if not self.is_drawing:
                    for pt in range(len(self.kept_cnts)):
                        M = cv2.moments(self.kept_cnts[pt])
                        if M["m00"]==0:M["m00"]=1
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        if self.found_pts[pt]!=10:
                            red_img = cv2.putText(red_img, str(self.found_pts[pt] + 1), (int(cX + 15), int(cY + 15)), cv2.FONT_HERSHEY_SIMPLEX, max(1,int(self.ratio*0.7)), (255, 0, 0), max(1,int(self.ratio * 2)))
                            red_img = cv2.line(red_img, (int(cX), int(cY)), (int(cX + 15), int(cY + 15)), (255,0,0), max(1,int(self.ratio * 1)))
                        else:
                            red_img = cv2.putText(red_img, str(self.found_pts[pt] + 1), (int(cX - 15), int(cY + 15)),cv2.FONT_HERSHEY_SIMPLEX, max(1, int(self.ratio * 0.7)), (255, 0, 0), max(1,int(self.ratio * 2)))
                            red_img = cv2.line(red_img, (int(cX), int(cY)), (int(cX - 10), int(cY + 10)), (255, 0, 0),max(1,int(self.ratio * 1)))

            else:
                im_to_show=np.copy(self.image)

            if self.is_drawing and cur_pos!="NA":
                im_to_show=cv2.circle(im_to_show,cur_pos,self.tool_size,(0,0,0),max(1,int(self.ratio*1)))


            best_ratio = max(self.Size[1] / (self.Canvas_for_video.winfo_width()),
                             self.Size[0] / (self.Canvas_for_video.winfo_height()))
            prev_final_width = self.final_width
            self.final_width = int(math.ceil(self.Size[1] / best_ratio))
            self.ratio = self.ratio * (prev_final_width / self.final_width)
            image_to_show = im_to_show[self.zoom_sq[1]:self.zoom_sq[3], self.zoom_sq[0]:self.zoom_sq[2]]
            image_to_show1 = cv2.resize(image_to_show,
                                        (self.final_width, int(self.final_width * (self.Size[0] / self.Size[1]))))
            self.image_to_show2 = Pil_imageTk.PhotoImage(image=PIL.Image.fromarray(image_to_show1))
            self.Canvas_for_video.create_image(0, 0, image=self.image_to_show2, anchor=NW)
            self.Canvas_for_video.config(width=self.final_width,height=int(self.final_width * (self.Size[0] / self.Size[1])))
        except Exception as e:
            print(e)
            pass



    def Zoom_in(self, event):
        #Zoom in the image
        self.new_zoom_sq = [0, 0, 0, 0]
        PX = event.x / ((self.zoom_sq[2] - self.zoom_sq[0]) / self.ratio)
        PY = event.y / ((self.zoom_sq[3] - self.zoom_sq[1]) / self.ratio)

        event.x = event.x * self.ratio + self.zoom_sq[0]
        event.y = event.y * self.ratio + self.zoom_sq[1]
        ZWX = (self.zoom_sq[2] - self.zoom_sq[0]) * (1 - self.zoom_strength)
        ZWY = (self.zoom_sq[3] - self.zoom_sq[1]) * (1 - self.zoom_strength)

        if ZWX > 25:
            self.new_zoom_sq[0] = int(event.x - PX * ZWX)
            self.new_zoom_sq[2] = int(event.x + (1 - PX) * ZWX)
            self.new_zoom_sq[1] = int(event.y - PY * ZWY)
            self.new_zoom_sq[3] = int(event.y + (1 - PY) * ZWY)

            self.ratio = ZWX / self.final_width
            self.zoom_sq = self.new_zoom_sq
            self.zooming = True
            self.show_img()

    def Zoom_out(self, event):
        #Zoom out from the image
        self.new_zoom_sq = [0, 0, 0, 0]
        PX = event.x / ((self.zoom_sq[2] - self.zoom_sq[0]) / self.ratio)
        PY = event.y / ((self.zoom_sq[3] - self.zoom_sq[1]) / self.ratio)

        event.x = event.x * self.ratio + self.zoom_sq[0]
        event.y = event.y * self.ratio + self.zoom_sq[1]

        ZWX = (self.zoom_sq[2] - self.zoom_sq[0]) * (1 + self.zoom_strength)
        ZWY = (self.zoom_sq[3] - self.zoom_sq[1]) * (1 + self.zoom_strength)

        if ZWX < self.Size[1] and ZWY < self.Size[0]:
            if int(event.x - PX * ZWX) >= 0 and int(event.x + (1 - PX) * ZWX) <= self.Size[1]:
                self.new_zoom_sq[0] = int(event.x - PX * ZWX)
                self.new_zoom_sq[2] = int(event.x + (1 - PX) * ZWX)
            elif int(event.x + (1 - PX) * ZWX) > self.Size[1]:
                self.new_zoom_sq[0] = int(self.Size[1] - ZWX)
                self.new_zoom_sq[2] = int(self.Size[1])
            elif int(event.x - PX * ZWX) < 0:
                self.new_zoom_sq[0] = 0
                self.new_zoom_sq[2] = int(ZWX)

            if int(event.y - PY * ZWY) >= 0 and int(event.y + (1 - PY) * ZWY) <= self.Size[0]:
                self.new_zoom_sq[1] = int(event.y - PY * ZWY)
                self.new_zoom_sq[3] = self.new_zoom_sq[1] + int(ZWY)

            elif int(event.y + (1 - PY) * ZWY) > self.Size[0]:
                self.new_zoom_sq[1] = int(self.Size[0] - ZWY)
                self.new_zoom_sq[3] = int(self.Size[0])
            elif int(event.y - PY * ZWY) < 0:
                self.new_zoom_sq[1] = 0
                self.new_zoom_sq[3] = int(ZWY)
            self.ratio = ZWX / self.final_width


        else:
            self.new_zoom_sq = [0, 0, self.image.shape[1], self.image.shape[0]]
            self.ratio = self.Size[1] / self.final_width

        self.zoom_sq = self.new_zoom_sq
        self.zooming = False
        self.show_img()




class Mainframe(Tk):
    # Launch the rest of animalTA
    def __init__(self):
        Tk.__init__(self)
        self.frame = Details_inter(self)
        self.frame.grid(sticky="nsew")



GWL_EXSTYLE = -20
WS_EX_APPWINDOW = 0x00040000
WS_EX_TOOLWINDOW = 0x00000080

root=Mainframe()
root.geometry("1250x720")
root.geometry("+100+100")

# all widgets will be here
# Execute Tkinter
root.mainloop()
