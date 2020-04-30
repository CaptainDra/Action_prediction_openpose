Human behavior detection and prediction by computer decision makers
Final report for SE740
Qingyang Long
April 27,2020
Abstract
Computer vision is popular nowaday as the development of cameras, computers and artificial intelligence. It can be applied in a wide range, and a lot of breakthroughs have been made by different universities. One of the most popular applications called Openpose is developed by Carnegie Mellon University, which can recognize humans from different body parts and link them together as skeleton diagrams. Using this method and combining the knowledge about robotic motion, we can analyze humans’ action state as robotic and make predictions about the future location of body parts. For this project, I used python as a programming environment, and import openpose tools of opencv in python. As a result, a demo of an application to recognize human’s static activities and dynamic activities and make prediction of hands location has been made by python. It is a sample to show how openpose works and can be developed to recognize more action in future.
Key word: computer vision, openpose, robotic motion, prediction.

Introduction
Nowaday, human recognition is a popular application for the development of computer science and artificial intelligence, such as human computer interaction, security and surveillance, shopping experience analysis and health care. People train some model of human activities for computers to let them recognize different actions. 

A common approach is training different pictures and videos about humans specific activities, and after training, the computer decision maker can recognize the specific action. It is similar to the common approach to recognize humans in pictures, train a lot of pictures of humans and then discern humans from pictures as a whole human.However, Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, Yaser Sheikh[1] of CMU update a method to recognize human’s different body parts at first and  then use part affinity fields to link body parts together as human’s skeleton diagrams. This is very enlightening that if computers can recognize humans different actions from the motion of body parts. For example, if a person’s hand is higher than his neck, the computer could recognize that as “Hand up.” 

In this report, I will give a brief description about the openpose at first, about how the computer recognizes humans as different parts and how computers link them together at first. Then, I will introduce the three different parts of my application demo. At last I will show what I can do in future and what I learned from the project. 

 Working principle of Openpose
2.1 General information about openpose 

Figure.1 difference between traditional method and openpose
Compared with the traditional method(recognize human as a whole entity) to recognize humans, it will be influenced by many factors, such as being covered or other human’s location or too many humans in one picture, and all of these shortages have been solved by openpose. This method creates a method called PAFs(Part Affinity Fields), to link a pair of body parts. 

Figure.2
As in figure.1, a picture or video as (a) is sent to the computer, it will be sent to two algorithms. The first one could get the confidence map to show body parts’ location as (b), and the second algorithm could get the PAFs of the picture as (c). Then the openpose combines the results of each algorithm, and we could get many pairs of body parts as (d) shows. At last we link them together and the result of human’s skeleton diagrams come out.

2.2  Body Part Association
The first step is to build confidence maps for part detection. At first we generate the confidence maps S*j,k from 2D keypoints. Assume xj,k be the position of body part j for person k(for one person the k is equal to 1) in the picture, σ is a control variable for the spread of peak, and the value at p is as,
                                                                    (1)
For one person in picture, we only need to consider the max of S*j,k  , and for multiple persons we need to find the max S*j,k  , for different k.  then we can mark the point in the picture as confidence maps.

The next step is using Part Affinity Fields to associate different body parts.

Figure.3
As the figure.2 shows, we have detected the body part as (a), and we have nine kinds of combinations. We need to get the correct answer instead of the wrong answer as the green line in (b). We have a value for each pair of combinations to show the probability of each pair is correct. Then we choose the best value as the result of this step. 

After we match every part in the figure.1, we could get the final result as figure 1.(e). Moreover, we could also get the position of every part of the body, which we can use in my project.

Current understanding and Motivation
3.1 Current understanding 
The theory of openpose really inspired me a lot, because it can recognize humans from a small part. I want to make something to predict what will happen in the future. As far as I read the usage of human recognizations, the recognizations can be used in many different situations. For example, there are many shopping malls with surveillance cameras. It can record what people are doing, and some people raise an idea about that if we can judge some danger action from humans before it happens. For example, if a person is glancing right and left, and always wants to see other’s pockets, the experienced police may judge the people as a thief. Can we repeat this process by an AI decision maker? 

3.2 Motivation
If I can make some predictions from some body parts motion but not consider the whole body’s motions? This is the motivation of my project. I hope to make a small step in this term and in future, I will keep doing this to make this as a whole application. 

Moreover, we have learned a lot about robotic motion in this semester. I want to make some practical applications with this knowledge. This project is also a good chance for me to combine the knowledge we learned with my coding skills.

Approach 
4.1 Plan to do
For this project, after I set up the openpose in my computer, I will try to finish the project as following three steps: (1) recognize static action of human, (2) recognize dynamic action of human, (3) make predictions about the dynamic action. I will focus on how to recognize dynamic action and make predictions in my project.

4.2 Static action
I made an easy judgement for static actions: to check the location of my neck and hands. If both hands are higher than the location of the person’s neck, then output “BOTH HANDS UP”, else if only one hand is higher than the location of the person’s neck, then output ”LEFT HAND UP” or “RIGHT HAND UP”.
4.3 Dynamic action
For dynamic action, we need two photos with a very short time interval as δt, then we need to make a judgement about whether the human in photos has moved. I assume the location of the key body part is S, and the change of S in these two pictures is δS. If the δS is not equal to zero, that means the key body part moved, then we could use the information to make judgments.

For example, for the static action, I am hand up, and for the next second, the location of my hand changes, but still higher than my neck. Then the algorithm could make a judgment that I am waving my hands.

However, sometimes I am not going to wave my hand but the algorithm still says that I am waving my hand. So I add a limitation for δS, only if the change of my hand’s location is bigger than a range then the algorithm will judge this as a dynamic action. 

4.4 Prediction for dynamic action
After the computer judged that I was waving my hand. Then it would make some predictions from the last two pictures. I made two modes for the prediction: (a) take the motion as a uniform linear motion, (b) take the motion as a rotation around a point.
4.4.1 Uniform linear motion

Figure.3
As the figure.3 shows, we have the location of the hand of the last picture as S1, and the location of the hand for now as S2. Then we could get the speed of my hand as v.
                                                              v=(S1-S2)/δt                                                         (2)
After we get the speed as v, we could get S* as the result of prediction.
   S*=S2+v*δt                                                         (3)

4.4.2 Uniform rotation

Figure.4
For this mode, I assume my hand is rotating around a point as a constant angular speed. As Figure.4 shows, we can get the location of the hand as Sh1 for last time and Sh2 for now, location of the elbow as Se1 for last time and Se2 for now. We assume the coordinates as (x,y) for each point.

Then we could get the location of the point(xo, yo) that my hand is rotating around as follows.
          k1  = (yh1 – ye1)/(xh1 – xe1)	   			(4)
k2  = (yh2 – ye2)/(xh2 – xe2)
b1  = ye1 - k1*xe1                                                                        
b2 = ye2 – k2*xe2
xo = (b2 - b1)/(k1 – k2)
yo = k1* xo + b1
After we get (xo, yo) , we can set this point as the new origin point for the coordinate system, and the hands location as (xe , ye),(xi, yi) for the location of hand for the last time and now. Then we could use the following equation to solve the angular speed problem.
                                  (5)
After using equation(5), we can get the solution of prediction for my hand for the next time.


 Result
5.1 Result for static action

                                                                            (b)
Figure.5
As the figure.5 shows, when both of my hands are higher than my neck, it will output “BOTH HANDS UP” as (a). If only one of my hands is higher than my neck, it will output”XXX HAND UP”, and for this sample, only my left hand is higher than my neck, so it outputs “LEFT HAND UP”.
5.2 Result for dynamic action and prediction
5.2.1 prediction method I

                                                                       (b)  
Figure.6
This is the result of dynamic action recognition and prediction. In the picture, the computer recognized that I had moved my hand, and it output the result as ”WAVE LEFT HAND” as (a). 

Also I apply my prediction rule I in that, the single red point in (a) means the prediction point for my hand for the next time.As figure.6(b) shows, I record my hand for a continuous period of time, and the algorithm keeps output the location of my neck, left hand and right hand. If my hand moves, it will output a sentence like “moving to:(x,y)”. As a result, for most of the time the prediction is similar to the location of my hand for the next time. So the result is accurate for waving hands.
5.2.2 prediction method II

(a)

(b)
Figure.7
For this case, I apply my prediction rule II, which is more complex than the first one. However, the result came out by this method is not as accurate as the first one. Especially for the y-coordinate, it will generate up and down deviation. As the figure(a) and (b) shows, the prediction is accurate for x-coordinate but a little higher than where my hand would be for the next time.

I try to make more results using this method, and analyze these results and compare these results with the first one. I think maybe the system of our body may be much more complex than I thought or our brain could control our body movement in a more straight way. 
5.2.3 Problem case

Figure.8

Figure.9
Because of the problem of light in my room, sometimes the computer cannot figure out my hand from the picture or camera as figure.9. Due to the problem the prediction may be also incorrect as figure.8, when I moved my hand slowly, the computer showed that I moved hand fastly(from out of screen moved into screen). 




6. Conclusions
For the result I got from my algorithm, I found that the easier method has better and more accurate results for my motion. However,it is not a universal solution, it may not be accurate for other motions such as run or walk or shake head,etc.Hence, it is still a good try for the application that I want to make, and I will work on it in future. 

In future, I will try to consider my upper arm prediction again. And try to recognize more action in our lives. Or do some research about that. 

For the project, I have learned about how to use opencv and openpose by python and also combine that with some knowledge we learned in this term. Moreover, I also know how to do a project. It was really important for me.

I will upload the code to my github:https://github.com/CaptainDra/Action_prediction_openpose/tree/master

I will finish the readme document later.











Reference:
[1] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, Yaser Sheikh, “OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,” in 2018.
[2] G. Hidalgo, Z. Cao, T. Simon, S.-E. Wei, H. Joo, and Y. Sheikh, “OpenPose library,” https://github.com/ CMU-Perceptual-Computing-Lab/openpose.
[3] P. F. Felzenszwalb and D. P. Huttenlocher, “Pictorial structures for object recognition,” in IJCV, 2005.
[4] ——, “Pictorial structures revisited: People detection and articulated pose estimation,” in CVPR, 2009.
[5] V. Belagiannis and A. Zisserman, “Recurrent human pose estimation,” in IEEE FG, 2017.
[6] X. Qian, Y. Fu, T. Xiang, W. Wang, J. Qiu, Y. Wu, Y.-G. Jiang, and X. Xue, “Pose-normalized image generation for person reidentification,” in ECCV, 2018.


