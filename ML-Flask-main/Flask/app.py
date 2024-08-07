from flask import Flask, request, render_template
import re

import tensorflow as tf

from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import numpy as np
app = Flask(__name__)
import random
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()


ps = PorterStemmer()



@app.route("/")
def home():
    return render_template("input.html")

@app.route("/submit" ,methods = ["POST"])
def index():
    textInput = request.form["textInput"]
    textInput=[textInput]
    X_new = vectorizer.transform(textInput)
    predictions = model.predict(X_new)
    mydict1={
0:["Nee Kosam from the movie Majnu","Em Sandeham Ledu from the movie Oohalu Gusagusalade","Yeduta Nilichindi Choodu from the movie Vaana","Oohalu Oorege Gaalanthaa from the movie Oohalu Gusagusalade","Vintunnavaa from the movie Yevade Subramanyam","Nee Jathaga from the movie Yevadu","Kani Penchina Ma Ammake from the movie Manam","Madhuram Madhuram from the movie Brindavanam","Yemo Yemo from the movie Yuddham Sharanam","Andamaina Anubhavam from the movie Seethamma Vakitlo Sirimalle Chettu","Tum Hi Ho from the movie Aashiqui 2","Channa Mereya from the movie Ae Dil Hai Mushkil","Tujh Mein Rab Dikhta Hai from the movie Rab Ne Bana Di Jodi","Jeene Bhi De from the movie Dil Sambhal Jaa Zara","Tum Mile from the movie Tum Mile","Tera Mera Rishta from the movie Awarapan","Kal Ho Naa Ho from the movie Kal Ho Naa Ho","Agar Tum Saath Ho from the movie Tamasha","Kabira from the movie Yeh Jawaani Hai Deewani","Teri Meri from the movie Bodyguard","Spring Day by BTS","Stigma by V (BTS)","Don't Cry by The Cross","Lonely by 2NE1","Through the Night by IU","Eyes, Nose, Lips by Taeyang","You Were Beautiful by DAY6","Last Dance by BIGBANG"]
,
1:["Vachinde from Fidaa","Top Lesi Poddi from Iddarammayilatho","Ammadu Let's Do Kummudu from Khaidi No. 150","Sir Osthara from Businessman","Crazy Feeling from Nenu Sailaja","Rama Rama from Srimanthudu","Aa Seetadevi Navvula from Kotha Janta","Super Machi from S/O Satyamurthy","Pakka Local from Janatha Garage","Chakori from Chakori","Happy from Desi Boyz","Badtameez Dil from Yeh Jawaani Hai Deewani","London Thumakda from Queen","Gallan Goodiyaan from Dil Dhadakne Do","Bom Diggy Diggy from Sonu Ke Titu Ki Sweety","Nashe Si Chadh Gayi from Befikre","Kar Gayi Chull from Kapoor & Sons","Gulaabo from Shaandaar","Nachde Ne Saare from Baar Baar Dekho","Badri Ki Dulhania from Badrinath Ki Dulhania","Dope by BTS","Fancy by TWICE","Love Scenario by iKON","TT by TWICE","Spring Day by BTS","Ko Ko Bop by EXO","Boom Boom by SEVENTEEN","Cheer Up by TWICE","Boy With Luv by BTS ft. Halsey"]
,
2:["Ee Jeevana Tarangalalo from Swathi Mutyam","Patala Pallakivai from Keeravani","Aakasamlo Aasala Harivillu from Nirnayam","Nuvvu Nenu Janta from Gangotri","Mounamelanoyi from Sagara Sangamam","Priyatama Neevachata Kusalama from Prema","Vintunnavaa from Ye Maaya Chesave","Swarabhishekam from Swarabhishekam","Kanne Pillavani from Kshana Kshanam","Palukaga from Annamayya","Tujh Mein Rab Dikhta Hai from Rab Ne Bana Di Jodi","Tujhse Naraz Nahi Zindagi from Masoom","Tera Mera Pyar Amar from Asli Naqli","Dil Hai Ke Manta Nahin from Dil Hai Ke Manta Nahin","Mera Dil Bhi Kitna Pagal Hai from Saajan","Mere Sapno Ki Rani Kab Aayegi Tu from Aradhana","Lag Jaa Gale from Woh Kaun Thi","Tum Aa Gaye Ho Noor Aa Gaya Hai from Aandhi","Ae Mere Humsafar from Baazigar","Lag Jaa Gale from Woh Kaun Thi","Spring Day by BTS","Love Scenario by iKON","You Are My Everything by Gummy","Through the Night by IU","Butterfly by BTS","Last Dance by BIGBANG","Don't Forget by Crush","Love, ing by Ben","Time Walking on Memory by Nell","I Will Go to You Like the First Snow by Ailee"],
3:["Adiga Adiga from Ninnu Kori","Choosi Chudangane from Chalo" ,"Neethone from Dhruva","Nuvvante Na Navvu from Krishnagadi Veera Prema Gaadha","Vachinde from Fidaa","Dhaari Choodu from Krishnarjuna Yudham","Chusi Chudangane from Chalo","Kukka Kavali from Temper","Dammu Dammu from Khaidi No. 150","Nee Kallalona from Jai Lava Kusa","Bolo Har Har Har from Shivaay","Jungle Jungle Baat Chali Hai from The Jungle Book","Kar Gayi Chull from Kapoor & Sons","Dilli Wali Girlfriend from Yeh Jawaani Hai Deewani","Badtameez Dil from Yeh Jawaani Hai Deewani","Karuppu Nerathazhagi from Kombani","Angry Mix from Mujhse Shaadi Karogi","Jai Ho from Slumdog Millionaire","Koi Kahe Kehta Rahe  from Dil Chahta Hai","Bhaag DK Bose from Delhi Belly", "Break Stuff by Limp Bizkit","Killing in the Name by Rage Against the Machine","Before I Forget by Slipknot","Down with the Sickness by Disturbed" ,"Last Resort by Papa Roach","Break My Stride by Matthew Wilder","You Oughta Know by Alanis Morissette","Lose Yourself by Eminem ","Breakdown by Tom Petty and the Heartbreakers" ,"Hit the Road Jack by Ray Charles ","MIC Drop by BTS" ,"Dope by BTS" ,"Monsterby EXO " ,"Fire by 2NE1" ,"War of Hormone by BTs","No More Dream by BTS" ,"Wolfby EXO" ,"I Am the Best by 2NE1","Cypher Pt.3: Killer by BTS","Mama by EXO"],

4:["Manasu Palike from the movie Pelli Choopulu","O Cheli from the movie Nenu Local","Adiga Adiga from the movie Ninnu Kori","Vintunnava from the movie Ye Maaya Chesave","Mari Mari from the movie Yuddham Sharanam","Vasthane Vasthane from the movie Nuvvu Naku Nachav","Vinnaana Vinnaana from the movie Sahasam Swasaga Sagipo","Idi Kalala Vunnadhe from the movie Ye Maya Chesave","Gundelothullo from the movie Geetha Govindam","Vinnane Vinnane from the movie Tholi Prema","Kal Ho Naa Ho from the movie Kal Ho Naa Ho","Ae Mere Humsafar from the movie Baazigar","Ae Zindagi Gale Laga Le from the movie Dear Zindagi","Tujh Mein Rab Dikhta Hai from the movie Rab Ne Bana Di Jodi","Lag Jaa Gale from the movie Woh Kaun Thi","Phir Kabhi from the movie M.S. Dhoni: The Untold Story","Tum Mile from the movie Tum Mile","Main Yahaan Hoon from the movie Veer-Zaara","Tum Jo Aaye from the movie Once Upon A Time in Mumbaai","Breathe Me by Sia","Mad World by Tears for Fears (covered by Gary Jules)","Anxiety by Jason Isbell and the 400 Unit","Let It Be by The Beatles","Don't Worry, Be Happy by Bobby McFerrin","Three Little Birds by Bob Marley","Lean On Me by Bill Withers","All of Me by John Legend","With a Little Help from My Friends by The Beatles","Somewhere Over the Rainbow by Israel Kamakawiwo'ole","Through the Night by IU","Love Scenario by iKON","Spring Day by BTS","You Were Beautifulby DAY6" ,"Every Day, Every Moment by Paul Kim","Time for the Moon Night by GFRIEND","Dear Name by IU","Palette by IU","Lonely by 2NE1"]
  }
    values = mydict1[predictions[0]]
    random_value = random.choice(values)
    return render_template("output.html", predict = random_value)

if __name__ == "__main__":
    app.run(debug=True,port = 1111)
