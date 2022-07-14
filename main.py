from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import tkinter.scrolledtext as st 
import cv2
from gtts import gTTS
from playsound import playsound
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def Input_Image():
    global img_path
    master.filename=filedialog.askopenfilename(parent=master,initialdir="/",title="choose your image",filetypes=(("png files","*.jpg"),("all files","*.*")))
    img=cv2.imread(master.filename)
    img_path=master.filename
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resized_image=cv2.resize(img,(400,350))
    image=Image.fromarray(resized_image)
    image=ImageTk.PhotoImage(image)
    panelA=Label(image=image)
    panelA.image=image
    panelA.place(x=100,y=115)

def Text_to_speech():
    Message = "Happy Vishu "
    speech = gTTS(text = Message)
    speech.save('DataFlair.mp3')
    playsound('DataFlair.mp3')



def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

    
def caption_generator():
    global img_path
    global description
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(weights= 'E:\\PROJECTS-2021\\SCEM-CS-1-IMAGE CAPTION\\workingcode\\xception_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False, pooling="avg")### pblm
    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    description = description.replace('start', '')
    text_area.delete("1.0", END)
    text_area.insert(INSERT,description) 
    text_area.configure(state ='normal')

   
def Text_to_speech():
    global description
    speech = gTTS(text = description)
    speech.save('DataFlair.mp3')
    playsound('DataFlair.mp3')
    os.remove("DataFlair.mp3")





master = Tk()
master.title('Image caption')
master.geometry('1366x768')
master.configure(background='gray')

menubar = Menu(master) 
filemenu = Menu(menubar,tearoff=0) 
menubar.add_cascade(label='Train Data', menu=filemenu) 
filemenu.add_command(label='Upload',command=None) 
filemenu.add_command(label='Train',command=None)  
helpmenu = Menu(menubar,tearoff=0)
menubar.add_cascade(label='Exit',menu=helpmenu)
master.config(menu=menubar) 

c1 = Canvas(master,bg='gray',width=1258,height=80)
c1.place(x=5,y=5)

l1 = Label(master,text='IMAGE CAPTIONING AND AUDIO GENERATOR',foreground="red",background='gray',font =('Verdana',20,'bold'))
l1.place(x=280,y=25)

c2 = Canvas(master,bg='gray',width=1258,height=400)
c2.place(x=5,y=90)

c3 = Canvas(master,bg='white',width=400,height=350)
c3.place(x=100 ,y=115)

text_area = st.ScrolledText(master,font="verdana 12") 
text_area.place(height=350,width=600,x=600,y=115)

c4 = Canvas(master,bg='gray',width=1258,height=160)
c4.place(x=5,y=495)
  
b1=Button(master,borderwidth=1,relief="flat",text='UPLOAD IMAGE',font="verdana 12 bold", bg="lightgray", fg="red",command=Input_Image)
b1.place(height=70,width=220,x=180,y=540)

b2=Button(master,borderwidth=1,relief="flat",text='CAPTION GENERATOR',font="verdana 12 bold", bg="lightgray", fg="red",command=caption_generator)
b2.place(height=70,width=220,x=605,y=540)

b3=Button(master,borderwidth=1,relief="flat",text='AUDIO GENERATOR',font="verdana 12 bold", bg="lightgray", fg="red",command=Text_to_speech)
b3.place(height=70,width=220,x=980,y=540)

mainloop()
 


