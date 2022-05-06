# -*- coding: utf-8 -*-
"""
Created on Fri May 06 00:03:18 2016

@author: jlarsch
"""

import numpy as np
import cv2
import geometry as geometry
import functions.ImageProcessor as ImageProcessor
import functions.splineTest as splineTest
import pickle
import datetime
import multiprocessing as mp
from contextlib import closing
import sys

#import itertools

def get_AnimalShapeParameters(path,trajectories,startFrame,nFrames,generateOutVideo=0):
    #obtain descriptive shape parameters of animals
    #typically starting from a video tracked (by idTracker), return to the video to extract further shape information
    #-direction based on iterative contour analysis: elipse fit & tail detection

    #features to extract:
    #   position between eyes: center_eyes
    #   center of chest&eyes: center_chest
    #   Line between center_eyes&center_chest: headVector
    #   Angle of headVector: headAngle
    #   Skeleton:listOfPoints
    #       tailStraight-ness
    video = cv2.VideoCapture(path)
    asp=[]

    
    for i in np.arange(startFrame,startFrame+nFrames):
        video.set(cv2.CAP_PROP_POS_FRAMES,i)
        img_frame_original = video.read()[1]
        
        try:
            gray=cv2.cvtColor(img_frame_original, cv2.COLOR_BGR2GRAY)
        except:
            print('error converting frame: '+str(i)+', using previous frame')
                
#        img_frame_original=cv2.cvtColor(img_frame_original, cv2.COLOR_BGR2GRAY)
        tc=trajectories[i,0,:]
        asp.append(AnimalShapeParameters_f(tc,gray,i,0,generateOutVideo))
        tc=trajectories[i,1,:]
        asp.append(AnimalShapeParameters_f(tc,gray,i,1,generateOutVideo))

    return asp
    
    
def storeOutputFFF(fff,theArgs,que): #add a argument to function for assigning a queue
    print('MULTIPROCESSING: Launching %s in parallel '%fff.func_name)
    que.put(fff(*theArgs)) #we're putting return value into queue
    
def vidSplit(path,trajectories,nframes=0,generateOutVideo=0):
    if nframes==0:
        nframes=trajectories.shape[0]
    #nframes=1000
    npro=mp.cpu_count()-1
    listOf_FuncAndArgLists=[]
    chunkSize=np.floor(nframes/npro)
    for i in range(npro):
        
        listOf_FuncAndArgLists.append([get_AnimalShapeParameters,path,trajectories,i*chunkSize,chunkSize,generateOutVideo])
        
    queues=[mp.Queue() for fff in listOf_FuncAndArgLists] #create a queue object for each function
    jobs = [mp.Process(target=storeOutputFFF,args=[funcArgs[0],funcArgs[1:],queues[iii]]) for iii,funcArgs in enumerate(listOf_FuncAndArgLists)]
    for job in jobs: job.start() # Launch them all
    #for job in jobs: job.join() # Wait for them all to finish
    # And now, collect all the outputs:
    asp_f=[queue.get() for queue in queues]
    currentTime=datetime.datetime.now()
    print(currentTime, ': saving data before cleanUp')
    pickleFile=path[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pickle'
    save_asp(pickleFile,asp_f)    
    
    asp=[]            
    asp.append(AnimalShapeParameters(asp_f,0))
    asp.append(AnimalShapeParameters(asp_f,1))
    
    asp=asp_cleanUp(asp)
    
    currentTime=datetime.datetime.now()
    print(currentTime, ': saving data after cleanUp')
    pickleFile=path[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pickle'
    save_asp(pickleFile,asp)
    
    return asp


def save_asp(fn,asp):
    with open(fn, 'w') as f:
        pickle.dump(asp, f)
    print(['data saved as ',fn])

def asp_cleanUp(asp):
    a=asp[0].fish_orientation_elipse
    b=asp[1].fish_orientation_elipse
    
    #distance between centroids
    #use as mask to flag collisions
    animal_dif=asp[0].centroidContour - asp[1].centroidContour
    dist=np.sqrt(animal_dif[:,0]**2 + animal_dif[:,1]**2)
    collision_frames=dist<1
    
    a[collision_frames]=np.nan
    b[collision_frames]=np.nan
    
    asp[0].fish_orientation_elipse[collision_frames]=np.nan
    asp[1].fish_orientation_elipse[collision_frames]=np.nan
    
    p1=asp[0].centroidContour
    p2=asp[1].centroidContour
#
#    p2b=np.roll(p2,experiment.sPair.shiftIndex[2],axis=0)
#    bb=np.roll(b,experiment.sPair.shiftIndex[2],axis=0)
#    
#    angle_centroid_connectb=geometry.get_angle_list(p1,p2b)
#    
#    #flip angles along x axis for consistent angle reference
#    acc_1to2b=np.mod(180-angle_centroid_connectb,360)
#    acc_2to1b=np.mod(180+acc_1to2b,360)
#    
#    #deviation of an1 from an2 centroid
#    dev1b=geometry.smallest_angle_difference_degrees(bb,acc_2to1b)  
#    dev0b=geometry.smallest_angle_difference_degrees(a,acc_1to2b)
#    
#    
    
    #compute delta heading as difference of heading to angle between centroids of the two animals
    angle_centroid_connect=geometry.get_angle_list(p1,p2)
    
    #flip angles along x axis for consistent angle reference
    acc_1to2=np.mod(180-angle_centroid_connect,360)
    acc_2to1=np.mod(180+acc_1to2,360)
    
    #deviation of an1 from an2 centroid
    asp[1].deviation=geometry.smallest_angle_difference_degrees(b,acc_2to1)  
    asp[0].deviation=geometry.smallest_angle_difference_degrees(a,acc_1to2)
    
    asp[0].baseline=acc_1to2
    asp[1].baseline=acc_2to1
    asp[0].allDist[collision_frames,:]=np.nan
    asp[1].allDist[collision_frames,:]=np.nan

    return asp
    
class AnimalShapeParameters(object):
        
    def __init__(self,asp_f,animal):
        self.trajectory=np.array([x.trajectory for x in [item for sublist in asp_f for item in sublist]][animal::2])
        self.threshold_elipse = asp_f[0][0].threshold_elipse
        self.threshold_skeleton=asp_f[0][0].threshold_skeleton
        
        self.trajectory=np.array([x.trajectory for x in [item for sublist in asp_f for item in sublist]][animal::2])
        self.skel_smooth_all=np.array([x.skel_smooth_all for x in [item for sublist in asp_f for item in sublist]][animal::2])
        #self.spine_angles_all=np.array([x.spine_angles_all for x in [item for sublist in asp_f for item in sublist]][animal::2])
        self.fish_orientation_elipse=np.array([x.fish_orientation_elipse for x in [item for sublist in asp_f for item in sublist]][animal::2])
        self.allDist=np.array([x.allDist for x in [item for sublist in asp_f for item in sublist]][animal::2])
        self.centroidContour=np.array([x.centroidContour for x in [item for sublist in asp_f for item in sublist]][animal::2])
        self.frame=np.array([x.i for x in [item for sublist in asp_f for item in sublist]][animal::2])
        if asp_f[0][0].generateOutVideo:
            self.frAll_rot=np.array([x.frAll_rot for x in [item for sublist in asp_f for item in sublist]][animal::2])
            self.frAll_rot_lg=np.array([x.frAll_rot_lg for x in [item for sublist in asp_f for item in sublist]][animal::2])

class AnimalShapeParameters_f(object):
        
    def __init__(self,trajectory,img_frame_original,i,animal,generateOutVideo=0):
        self.trajectory=trajectory
        self.threshold_elipse = 185
        self.threshold_skeleton=240
        #ROI size around animal 
        #with animal head centerd in ROI, ROI must not clip animal in any orientation.
        cropSize=300
        self.cropSize=cropSize
        
        self.skel_smooth_all=np.zeros((30,2))
        
        self.centroidContour=np.zeros(2)
        self.allDist=np.zeros(30)
        
        self.fish_orientation_elipse=0
        self.good_skel=1
        self.generateOutVideo=generateOutVideo
        self.i=i
        self.animal=animal

        if generateOutVideo:
            self.frAll_rot=np.zeros((cropSize,cropSize),dtype='uint8')
            self.frAll_rot_lg=np.zeros((900,900,3),dtype='uint8')
            img_rot_binary_lg=np.zeros((900,900,3),dtype='uint8')
        
        #if np.mod(i,100)==0:
        #    print i,"         \r",
        if np.mod(i,10000)==0:
            currentTime=datetime.datetime.now()
            print(currentTime,'animal: ',animal, ' frame: ',i)
            sys.stdout.flush()
            
        #process sub region around fish
        currCenter=geometry.Vector(*self.trajectory)
        img_crop=ImageProcessor.crop_zero_pad(img_frame_original,currCenter,self.cropSize)
        
        #currently, best strategy for orientation seems to be a sequential process:
        #   1: Use a robust 0-180 degree measure to find major orientation of the fish shape
        #       *-a) fit ellipse on (low) rigid body threshold contour, ellipse angle is robust chest-eye orientation
        #       -b) use image moments
        #   2: use second approach to distinguish head-tail
        #       *-a) find tail in polygonFit on low threshold contour, find tail-center direction
        #       -b)  use hull contour instead of polygon to find tail tip [NOT robust when tail bends - deleted]
        #       -c) find darkest pixel, assume it is eye, find center-eye direction [NOT robust, sometimes chest darker than eye]
        #   3: (not yet implemented) refine elipse orientation using eye position
        
        
        # 1: find robust 0-180 major axis orientation
        # use a low threshold contour that includes the rigid fish body but not the flexible tail
        
        contour_fish_elipse,centerOfMass_elipse = self.get_fish_contour(img_crop.copy(),self.threshold_elipse)
        self.fish_orientation_elipse=self.get_fish_orientation(contour_fish_elipse,centerOfMass_elipse)
        
        contour_fish_skel, centerOfMass_skel = self.get_fish_contour(img_crop.copy(),self.threshold_skeleton)
        #centroid of contour related back to un-cropped image
        try:        
            self.centroidContour=np.add(self.trajectory,np.subtract([centerOfMass_skel.x,centerOfMass_skel.y],[151,149]))
        except:
            print('no center of mass for frame: ',i)
            
        if not np.isnan(self.fish_orientation_elipse):
        
            #generate rotation cancelled image
            #translate contour centroid to image center
        
            #translation
            M_trans = np.float32([[1,0,self.cropSize/2-centerOfMass_elipse[0]],[0,1,self.cropSize/2-centerOfMass_elipse[1]]])
            animalCenter=geometry.Vector(self.cropSize/2,self.cropSize/2)
            img_trans = 255-cv2.warpAffine(255-img_crop,M_trans,img_crop.shape)
            
            #rotation
            M_rot = cv2.getRotationMatrix2D((self.cropSize/2,self.cropSize/2),self.fish_orientation_elipse,1)
            img_rot = 255-cv2.warpAffine(255-img_trans,M_rot,img_trans.shape)
            
            
            #mask away everything other than current animal
            contour_skel, centerOfMass = self.get_fish_contour(img_rot.copy(),self.threshold_skeleton,animalCenter)
            
            mask = np.ones(img_crop.shape[:2], dtype="uint8") * 0
            cv2.drawContours(mask, contour_skel, -1, 255, -1)
            mask=255-mask
            img_rot_binary=ImageProcessor.to_binary(mask.copy(), self.threshold_skeleton)
            
            #get skeleton and segment angles from binary image
            skel_angles,skel_smooth,skel,ep=splineTest.return_skeleton(img_rot_binary)
                
            self.spine_angles_all=skel_angles
            if skel_smooth is not None:
                skel_smooth=np.roll(skel_smooth,1,axis=1)
                if self.generateOutVideo:
                    img_rot_binary_lg=cv2.resize(img_rot_binary,(900,900),interpolation = cv2.INTER_CUBIC)
                    cv2.polylines(img_rot_binary,np.int32(skel_smooth).reshape((-1,1,2)),True,(100,100,100))
                    
                    img_rot_binary_lg=cv2.cvtColor(img_rot_binary_lg,cv2.COLOR_GRAY2BGR)
                    cv2.polylines(img_rot_binary_lg,np.int32(skel_smooth).reshape((-1,1,2))*3,True,(255,0,0),thickness = 3)
                for j in range(len(skel_smooth)):
                    try:
                        self.allDist[j]=geometry.distance_point_line(skel_smooth[j],skel_smooth[0],skel_smooth[-1])
                    except:
                        self.allDist[0]=np.nan                     
                
            else:
                self.good_skel=0
                self.allDist[0]=np.nan
                #print ['error extracting skeleton in frame: ',i]
                

            self.skel_smooth_all=skel_smooth
        
                
        if self.generateOutVideo:
            self.frAll_rot[:,:]=img_rot_binary   
            self.frAll_rot_lg=img_rot_binary_lg

    def get_tail_tip_polygon(self,contour,epsilonFactor=0.02):         
    
        #use polygon around contour to find tail tip as sharpest angle      
        epsilon = epsilonFactor*cv2.arcLength(contour[0],True)
        poly = cv2.approxPolyDP(contour[0],epsilon,True)
        
        #get inner angles of all contour points
        poly_angles=geometry.get_contour_inner_angles(poly)
        tail_tip=poly[np.argmin(poly_angles)]
        
        return tail_tip
    

    
    def get_fish_ellipse_angle(self,contour):

        elipse_fit=cv2.fitEllipse(contour[0])
        elipse_orientation=np.mod(90+elipse_fit[2],180)
        return elipse_orientation
        
        
    def get_fish_contour(self,img,threshold,animalCenter=None):
        img_binary = ImageProcessor.to_binary(img.copy(), threshold)            
        im_tmp2,contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #only keep contour containing the trajectory coordinate
        #this would be the center of the cropped input image
        if animalCenter==None:
            img_center=geometry.Vector(img.shape[0]/2,img.shape[1]/2)
            animalCenter=img_center
            
        contour = ImageProcessor.get_contour_containing_point(contours,animalCenter)
        
        #contour = ImageProcessor.get_biggest_n_contours(contours, 1)
        if contour is not None:
            centerOfMass = ImageProcessor.get_contours_centroid(contour)
        else:
            centerOfMass= None
        return contour, centerOfMass
            
    def get_fish_orientation(self,contour_fish_elipse,centerOfMass):

        if contour_fish_elipse is not None:            
            fish_orientation_elipse = self.get_fish_ellipse_angle(contour_fish_elipse)
        
        # orient major axis using head-tail differences
        #find tail tip using polygon fit
            tail_tip_polygon=self.get_tail_tip_polygon(contour_fish_elipse)
        
        #get angle from tail to center of mass            
            tail_to_center_angle= np.mod(180-geometry.Vector.get_angle(geometry.Vector(*tail_tip_polygon[0]),centerOfMass),360)

        #use elipse orientation, flip to approximately match tail-center angle (within 60 degrees)
        #tail-center angle and elipse angle should be off by only a couple of degrees or swapped by 180 degrees
            if np.abs(tail_to_center_angle-fish_orientation_elipse)>60:
                if np.abs(tail_to_center_angle-fish_orientation_elipse)<300:
                    fish_orientation_elipse=np.mod(fish_orientation_elipse+180,360)
            
            return fish_orientation_elipse
            
        else:
            
            return np.nan

    

                
#            contours_eyes,good_eyes=find_good_eyes(img_crop,threshold_eyes,max_eye_distance,threshold_step,threshold_steps)
#            
#            if good_eyes:
#                
#                # Get the center point of these contours (the center point between the eyes)
#                center_eyes = ImageProcessor.get_contours_centroid(contours_eyes)
#                
#                # ***************************************************************
#                #                       Detecting the chest                     *
#                # ***************************************************************
#                #
#                # The next steps tries to detect the chest of the fish
#                # This step can only be applied if the eyes were detected.
#
#                # get the sharped image of the original frame (don't use the previous processed image because the threshold differs between the eyes and the chest)
#                img_finding_chest = img_crop.copy()
#        
#                # Convert the image into a binary image (black and white) using the threshold of the chest.
#                img_binary_chest = ImageProcessor.to_binary(img_finding_chest, threshold_chest)
#        
#                # Dilate the image (increase the white pixels, so the spaces between them will be reduced).
#                img_dilated_chest = cv2.dilate(img_binary_chest, kernel, dilate_iteration_chest)
#                #img_dilated_chest=img_binary_chest.copy()
#                
#                # Get all the contours in the image
#                im_tmp2,contours_chest, hierarchy = cv2.findContours(img_dilated_chest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        
#                # Keep the biggest two contours (they represent the chest and both eyes combined in one blob)
#                contours_chest = ImageProcessor.get_biggest_n_contours(contours_chest, 2)
#        
#                # Get the center point of these contours (the center point between the chest and the eyes)
#                center_chest = ImageProcessor.get_contours_centroid(contours_chest)
#        
#                # ***************************************************************
#                #                Calculate the direction of the fish.           *
#                # ***************************************************************
#                #
#                # The direction is the vector from the chest to the eyes.
#        
#                # Calculate the direction of the fish (from the chest toward the center of the eyes)
#                headVector = center_eyes.__sub__(center_chest)      
#                fish_angle = geometry.Vector.get_angle(center_chest, center_eyes)
#                dirAll[i]=fish_angle
#                
#                
#                if center_eyes:
#                    #cv2.circle(img_frame_original, (center_eyes.x + fish_region.top_left.x, center_eyes.y + fish_region.top_left.y), 5,
#                    #   (0, 0, 255), 2)
#                    cv2.circle(img_frame_original, (center_eyes.x, center_eyes.y), 2,(0, 0, 255), 1)
#                    cv2.circle(img_frame_original, (center_chest.x, center_chest.y), 2,(0, 0, 255), 1)
#                    cv2.imshow('Frame', img_frame_original)
#                    frAll.append(img_frame_original)
                
                

    def find_good_eyes(img,start_threshold,max_eye_distance=5,threshold_step=0.1,threshold_steps=10):
        #want:
        #2 blobs
        #appropriate distance
        good_eyes=False
        threshold_eyes=start_threshold
        contours_eyes,hierarchy,good_eyes=find_eyes(img.copy(),threshold_eyes,max_eye_distance)
        #expect two contours close to each other (eye distance)
          
        if contours_eyes.__len__() >2 or ((contours_eyes.__len__() == 2) and (good_eyes==False)):
            step=0
            while contours_eyes.__len__() >2 or ((contours_eyes.__len__() == 2) and (good_eyes==False)):
                #threshold too low, already including chest.
                #--> increase threshold by threshold_step
                threshold_eyes=int(threshold_eyes - threshold_eyes*threshold_step)
                step += 1
                if step > threshold_steps:
                    break
                contours_eyes,hierarchy,good_eyes=find_eyes(img.copy(),threshold_eyes,max_eye_distance)
        
        if contours_eyes.__len__() <2:
            step=0
            while contours_eyes.__len__() <2:
                #threshold too high, only getting one eye.
                #--> lower threshold by threshold_step
                threshold_eyes=int(threshold_eyes + threshold_eyes*threshold_step)
                step += 1
                if step > threshold_steps:
                    break
                contours_eyes,hierarchy,good_eyes=find_eyes(img.copy(),threshold_eyes,max_eye_distance)
         
        return contours_eyes,good_eyes
            
    
        
    def find_eyes(img,start_threshold,max_eye_distance):
        img_binary_eyes = ImageProcessor.to_binary(img.copy(), start_threshold)
        good_eyes = False
        # Dilate the image (increase the white pixels, so the spaces between them will be reduced).
        #img_dilated_eyes = cv2.erode(img_binary_eyes, kernel, dilate_iteration_eyes)
        #img_dilated_eyes=img_binary_eyes.copy()
        # Get all the contours in the image
        im_tmp,contours_eyes, hierarchy = cv2.findContours(img_binary_eyes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #remove very small contours (noise)            
        contours_eyes=remove_contours_smaller_than(contours_eyes,0.1)
        
        #if finding 3 contours, chest probably included.
        #select the 2 that are closest to each other
        
        
        
        if contours_eyes.__len__() ==2:
            #find centroids of each contour
            center_each_eye=get_each_contours_centroid(contours_eyes)
            #calculate distance of the contours
            center_distance=center_each_eye[0].__sub__(center_each_eye[1]).norm()
            #distance should be within bounds [this should be rather robust]
            if center_distance < max_eye_distance:
                good_eyes = True
                
            #else:
                
                #for now: give up on this frame
                
                #eyes might have been combined into one blob and chest was the other blob?
                #--> reduce threshold to separate the eyes
            
                #OR one eye was below threshold?
        
        
        return contours_eyes,hierarchy,good_eyes
        
if __name__ == '__main__':
    mp.freeze_support()