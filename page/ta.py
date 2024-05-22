import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten,Input
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display, HTML

sdir=r'/content/drive/MyDrive/TA 2/Classification/TA angelia ldd_class.v4i.folder/train'

filepaths=[]
labels=[]
classlist=os.listdir(sdir)
for klass in classlist:
    classpath=os.path.join(sdir,klass)
    if os.path.isdir(classpath):
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            filepaths.append(fpath)
            labels.append(klass)
Fseries= pd.Series(filepaths, name='filepaths')
Lseries=pd.Series(labels, name='labels')
df=pd.concat([Fseries, Lseries], axis=1)
print (df.head())
print (df['labels'].value_counts())

train_split = 0.7
validation_split = 0.2
test_split = 0.1

# Split data into train and remaining data
train_df, remaining_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)

# Split remaining data into test and validation data
test_df, valid_df = train_test_split(remaining_df, test_size=test_split / (test_split + validation_split), shuffle=True, random_state=123)

print('train_df length:', len(train_df), 'test_df length:', len(test_df), 'valid_df length:', len(valid_df))

height=128
width=128
channels=3
batch_size=64

img_shape=(height, width, channels)
img_size=(height, width)
length=len(test_df)
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]
test_steps=int(length/test_batch_size)
print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps)

# augmentasi untuk semua dataset, termasuk data uji
gen = ImageDataGenerator( rescale=1./255,
    rotation_range=15,      # rotasi gambar sejauh 40 derajat
    # width_shift_range=0.1,  # geser gambar sejauh 20% dari lebar
    # height_shift_range=0.1, # geser gambar sejauh 20% dari tinggi
    # shear_range=0.2,        # shear transformation
    # zoom_range=0.2,         # zoom in/out sejauh 20%
    # horizontal_flip=True,   # flip horizontal
    # fill_mode='nearest'     # metode pengisian piksel yang hilang
)

# generator untuk data latih
train_gen = gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)

# generator untuk data validasi
valid_gen = gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)

# generator untuk data uji
test_gen = gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                   color_mode='rgb', shuffle=False, batch_size=test_batch_size)

# mengambil daftar kelas
classes = list(train_gen.class_indices.keys())
print(classes)
class_count = len(classes)

def show_image_samples(gen):
    test_dict=test_gen.class_indices
    classes=list(test_dict.keys())
    images,labels=next(gen) # get a sample batch from the generator
    plt.figure(figsize=(20, 20))
    length=len(labels)
    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image=images[i]
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='brown', fontsize=12)
        plt.axis('off')
    plt.show()

base_model=tf.keras.applications.VGG16(include_top=False, weights="imagenet",input_tensor=Input(shape=(128,128,3)))

base_model.summary()

base_model.trainable = False

model_name='CNN VGG16'
print("Building model with", base_model)
model = tf.keras.Sequential([
            # Note the input shape is the desired size of the image 128x128 with 3 bytes color
            # This is the first convolution
            base_model,
            tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu', strides=1), #filter=32 adalah jumlah bobotnya yang akan dikali dengan chanel dan ditambah dengan biasnya
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2), #melakukan pooling
            tf.keras.layers.Dropout(rate=0.5), #dropout itu teknik mengurangi overfitting

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4, activation='softmax') #4 artinya digunakan 4 unit neuron karena ada 4 classs, kemudian dilakukan aktivasi
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='categorical_crossentropy', metrics='accuracy')

model.summary()

epochs =50

history=model.fit(x=train_gen, epochs=epochs, validation_data=valid_gen)

def print_in_color(txt_msg,fore_tupple,back_tupple,):
    #mencetak text_msg dalam warna latar depan yang ditentukan oleh fore_tupple dengan latar belakang yang ditentukan oleh back_tupple
    #text_msg adalah teksnya, fore_tupple adalah tupple warna latar depan (r,g,b), back_tupple adalah tupple latar belakang (r,g,b)
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m'
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True) # returns default print color to back to black
    return

def tr_plot(tr_data, start_epoch): #membuat plot dari data pelatihan dan validasi dari model pembelajaran mesin

    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    #mengekstrak nilai akurasi dan kerugian (loss) dari data pelatihan (tr_data) dan validasi.

    Epoch_count=len(tacc)+ start_epoch #Menghitung jumlah total epoch berdasarkan data pelatihan dan menambahkannya dengan epoch awal
    Epochs=[] #Inisialisasi daftar kosong untuk menyimpan nomor epoch.

    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1) #Loop untuk menambahkan nomor epoch ke dalam daftar

    index_loss=np.argmin(vloss)
    val_lowest=vloss[index_loss]
    #Mencari indeks epoch dengan nilai loss validasi terendah dan menyimpannya
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    #Mencari indeks epoch dengan nilai loss validasi tertinggi dan menyimpannya
    plt.style.use('fivethirtyeight') #Mengatur gaya plot menggunakan gaya 'fivethirtyeight' gaya plot yang dicirikan oleh tampilan yang bersih, modern, dan profesional.
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    #Membuat label untuk titik-titik pada plot yang menunjukkan nilai kerugian validasi terendah dan akurasi validasi tertinggi.

    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8)) #Membuat gambar dan satu set sumbu untuk plot, dengan dua subplot sejajar dengan lebar total 20 inci dan tinggi 8 inci.

    axes[0].plot(Epochs,tloss, 'r', label='Training loss') #Membuat plot untuk kerugian pelatihan
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' ) #Membuat plot untuk kerugian validasi
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    #Menambahkan titik pada plot yang menunjukkan nilai kerugian validasi terendah dan akurasi validasi tertinggi.

    #set title untuk Menambahkan judul, label sumbu x, dan label sumbu y untuk subplot pertama dan kedua.
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)

    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout #Mengatur layout plot agar tidak tumpang tindih.
    #plt.style.use('fivethirtyeight')
    plt.show() #tampilkan

    def print_info( test_gen, preds, print_code, save_dir, subject ):
    class_dict=test_gen.class_indices #Membuat dictionary yang mengaitkan nama kelas dengan indeks kelas.
    labels= test_gen.labels #Mendapatkan label yang sebenarnya dari data pengujian.
    file_names= test_gen.filenames #Mendapatkan nama file gambar dari data pengujian.

    error_list=[]
    true_class=[]
    pred_class=[]
    prob_list=[]
    new_dict={}
    error_indices=[]
    y_pred=[]
    #Menginisialisasi list dan dictionary yang akan digunakan untuk menyimpan informasi tentang kesalahan klasifikasi.

    for key,value in class_dict.items(): #Membuat dictionary baru yang memetakan indeks kelas ke nama kelas.
        new_dict[value]=key             # dictionary {int dari class number: string dari class name}
    # store new_dict as a text fine in the save_dir

    classes=list(new_dict.values())     # Membuat string daftar nama kelas.
    dict_as_text=str(new_dict)
    dict_name= subject + '-' +str(len(classes)) +'.txt'
    dict_path=os.path.join(save_dir,dict_name)
    with open(dict_path, 'w') as x_file:
        x_file.write(dict_as_text)
    #Menyimpan dictionary kelas ke dalam file teks di direktori yang ditentukan.

    errors=0 # Inisialisasi variabel untuk menghitung jumlah kesalahan klasifikasi.

    for i, p in enumerate(preds): #Iterasi melalui hasil prediksi untuk setiap gambar.
        pred_index=np.argmax(p) #Mendapatkan indeks kelas dengan probabilitas prediksi tertinggi.
        true_index=labels[i]  # labels are integer values
        if pred_index != true_index: # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors=errors + 1
            #Menyimpan informasi tentang kesalahan klasifikasi.

        y_pred.append(pred_index) #Menambahkan prediksi model ke dalam list y_pred
    if print_code !=0: #Memeriksa apakah pengguna meminta untuk mencetak informasi tentang kesalahan klasifikasi.
        if errors>0:
            if print_code>errors:
                r=errors
            else:
                r=print_code
            msg='{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class' , 'True Class', 'Probability')
            print_in_color(msg, (0,255,0),(55,65,80))
            for i in range(r):
                split1=os.path.split(error_list[i])
                split2=os.path.split(split1[0])
                fname=split2[1] + '/' + split1[1]
                msg='{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i],true_class[i], ' ', prob_list[i])
                print_in_color(msg, (255,255,255), (55,65,60))
                #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg='With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0,255,0),(55,65,80))

    if errors>0: #Memeriksa apakah ada kesalahan klasifikasi yang terjadi.
        plot_bar=[]
        plot_class=[]
        # Inisialisasi list untuk menyimpan informasi tentang jumlah kesalahan per kelas.

        for  key, value in new_dict.items(): #Iterasi melalui dictionary nama kelas dan indeks kelas.
            count=error_indices.count(key) #Menghitung jumlah kesalahan yang terjadi untuk kelas tertentu.
            if count!=0: #Memeriksa apakah ada kesalahan yang terjadi untuk kelas tersebut.
                plot_bar.append(count)
                plot_class.append(value)   # Menyimpan jumlah kesalahan dan nama kelas ke dalam list.
        fig=plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        #Membuat dan mengonfigurasi plot untuk menampilkan jumlah kesalahan per kelas.

        for i in range(0, len(plot_class)): #Iterasi melalui daftar kelas yang memiliki kesalahan.
            c=plot_class[i]
            x=plot_bar[i] #Mendapatkan nama kelas dan jumlah kesalahan untuk kelas tertentu.
            plt.barh(c, x, ) #Menambahkan bar horizontal ke dalam plot.
            plt.title( ' Errors by Class on Test Set')
    y_true= np.array(labels)
    y_pred=np.array(y_pred) #Mengubah label yang sebenarnya dan prediksi model menjadi array numpy.

    if len(classes)<= 30: #Memeriksa apakah jumlah kelas kurang dari atau sama dengan 30.
        # Membuat confusion matrix
        cm = confusion_matrix(y_true, y_pred )
        length=len(classes)
        if length<8:
            fig_width=8
            fig_height=8
        else:
            fig_width= int(length * .5)
            fig_height= int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length)+.5, classes, rotation= 90)
        plt.yticks(np.arange(length)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes) #Menghitung laporan klasifikasi yang berisi precision, recall, dan F1-score untuk setiap kelas.
    print("Classification Report:\n----------------------\n", clr)

    tr_plot(history,0) #Memanggil fungsi tr_plot untuk membuat plot dari histori pelatihan (history). Argumen kedua, yaitu 0, mengindikasikan bahwa pelatihan dimulai dari epoch ke-0.
save_dir=r'./' #Mendefinisikan direktori penyimpanan (save_dir) untuk menyimpan model yang telah dilatih.
subject='pest' #Mendefinisikan subjek atau topik dari model yang akan disimpan.
acc=model.evaluate( test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1]*100
#Menggunakan metode evaluate pada model untuk menghitung akurasi pada set data pengujian (test_gen). Hasilnya dikalikan dengan 100 untuk mendapatkan nilai dalam persen.
msg=f'accuracy on the test set is {acc:5.2f} %' #embuat pesan yang berisi informasi akurasi pada set data pengujian.
print_in_color(msg, (0,255,0),(55,65,80)) #Mencetak pesan ke layar dengan warna teks hijau di latar belakang abu-abu gelap.
save_id=str (model_name +  '-' + subject +'-'+ str(acc)[:str(acc).rfind('.')+3] + '.h5')
#Mendefinisikan nama file untuk model yang akan disimpan. Nama file terdiri dari nama model, subjek, dan nilai akurasi yang dibulatkan menjadi dua angka di belakang koma.
save_loc=os.path.join(save_dir, save_id) #endefinisikan lokasi penyimpanan model dengan menggabungkan direktori penyimpanan (save_dir) dengan nama file.
model.save(save_loc) #Menyimpan model ke lokasi yang telah ditentukan.

print_code=0 #Ini menunjukkan bahwa tidak akan ada pesan error yang dicetak.
preds=model.predict(test_gen) #Menggunakan model untuk melakukan prediksi pada data pengujian (test_gen) dan menyimpan hasil prediksi dalam variabel preds.
print_info( test_gen, preds, print_code, save_dir, subject )

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# masukkan path dari file yang akan di prediksi
image_path = "/content/drive/MyDrive/TA 2/data test/bercak.jpg"

# Load model training nya  yang telah di simpan
loaded_model = tf.keras.models.load_model("/content/CNN VGG16-pest-75.36.h5")

# Load preprocess yang akan dilakukan
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to [0,1]

# Buat predictions
predictions = loaded_model.predict(img_array)

# buat kelas prediksinya
predicted_class = np.argmax(predictions)

# Map index kelas prediksinya ke actual class label
class_dict = {0: 'Daun Bercak', 1: 'Daun Keriting', 2: 'Daun Kuning', 3: 'Daun Sehat'}
predicted_label = class_dict[predicted_class]

# Print tprediksi
print(f"The predicted class is: {predicted_label}")
