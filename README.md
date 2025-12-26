
<h1 align="center">Eliminare Background In Timp Real</h1>
<h4 align="right">Stefan Pavel</h4>
<h4 align="right">Cristian-Stefan Burlacu</h4>
<h2 align="left">Descriere Proiect</h2>
<p align="left">
  Un sistem automat care captează video în timp real, segmentează persoana folosind o rețea neuronală, 
  și înlocuiește fundalul la alegerea utilizatorului.
</p>
<h2 align="left">Proiecte Similare</h2>

| Nr. | Autor(i) / An                                          | Titlul articolului / proiectului                                      | Aplicație / Domeniu                     | Tehnologii utilizate                   | Metodologie / Abordare                                    | Rezultate                                                  | Limitări                                                  | Comentarii suplimentare                              |
|-----|-------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------|--------------------------------------|----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|-------------------------------------------------------|
| 1   | Cioppa, Van Droogenbroeck, Braham (2020)              | Real-Time Semantic Background Subtraction                             | Supraveghere video                    | CNN, rețele neuronale                | Subtracție semantică combinată cu algoritmi clasici       | Performanță în timp real cu acuratețe bună                 | Sensibil la schimbări bruște ale iluminării               | Metodă eficientă pentru scene statice                   |
| 2   | Wang, Gao, Li, Lin, Cheng, Sun (2020)                                     | Removing the Background by Adding the Background                      | Reprezentare video auto-supervizată  | Deep learning                       | Învățare auto-supervizată pentru robustețe la fundal      | Model robust la variații de fundal                          | Complexitate ridicată a antrenării                         | Aplicații în analiza video neetichetată                 |
| 3   | Lin, Ryabtsev, Sengupta,Kemelmacher-Shlizerman (2020)                                     | Real-Time High-Resolution Background Matting                         | Editare video, realitate augmentată  | Rețele neuronale convoluționale     | Rețele duale pentru matting la rezoluție înaltă           | Procesare 4K la 30fps, calitate ridicată                   | Necesită GPU puternic                                    | Ideal pentru aplicații în timp real de înaltă rezoluție |
| 4   | Bahri, Ray (2023)                                     | Weakly Supervised Realtime Dynamic Background Subtraction             | Monitorizare dinamică                 | Învățare slab-supervizată            | Abordare slab-supervizată pentru fundal dinamic           | Performanță bună fără etichete la nivel de pixel           | Limitat în condiții extreme de iluminare                  | Bine adaptat pentru medii dinamice                       |
| 5   | Shahi, Li (2023)                                      | Deep Learning for Background Replacement in Video Conferencing       | Conferințe video                     | U-Net, MobileNet, ConvLSTM            | Modele multiple pentru înlocuire fundal                    | Evaluare comparativă a metodologiilor, rezultate bune       | Necesită date bine etichetate                             | Folosit în aplicații comerciale și open-source           |
<h2 align="left">Schema Bloc A Proiectului Si Descriere Module</h2>

                  +-----------------+
                  |  Captură video  |  ------------> Capturează în timp real stream-ul video de la cameră.      
                  +--------+--------+                Acesta furnizează cadrele (frame-urile) necesare procesării ulterioare.
                           |
                           v
                  +-----------------+              Pregătește cadrele video pentru segmentare: redimensionare, 
                  |   Preprocesare  |  ------------>  normalizare, conversie în formatul necesar rețelei neuronale. 
                  +--------+--------+
                           |
        --------------------
        |   +-------------------------------+                   
        |   |  Antrenare (Rețea neuronală)  |  ----->  Antreneaza o retea Neuronala pe o baza de date.
        |   +---------------+---------------+
        |                 |
        |                 v
        | +-------------------------------+         Aplică o rețea neuronală antrenată pentru segmentarea semantică,           
        +>|  Segmentare (Rețea neuronală) |  ----->  identificând silueta persoanei în fiecare cadru.
          +---------------+---------------+
                          |
                          v
                  +-----------------+                 Rafinează masca obținută de la rețea: elimină zgomotul, netezește marginile 
                  |  Postprocesare  |  ------------>  și aplică eventuale filtre.
                  +--------+--------+
                           |
                           v
                  +---------------------+             Înlocuiește fundalul original cu o imagine aleasă de utilizator, 
                  |  Background Replace |  ----------> păstrând doar persoana segmentată în prim-plan.
                  +--------+------------+
                           |
                           v
                  +------------------+                Afișează în timp real cadrul modificat, cu fundalul înlocuit, 
                  | Afișare rezultat |  ------------> într-o fereastră sau interfață grafică.
                  +--------+---------+
                           ^
                           |
              +------------+-------------+
              |     Control utilizator   |--------->Permite utilizatorului să schimbe fundalul sau să pornească/oprească procesul.
              +--------------------------+


