import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback # Hata izlerini görmek için eklendi
import requests # Model dosyasını URL'den indirmek için eklendi
import re # 'confirm' parametresini ayıklamak için eklendi

st.set_page_config(layout="wide")

st.title('Enerji Tüketimi Tahmin Uygulaması')

# --- Fonksiyon: Gerekli Dosyaların Yüklendiğinden Emin Olma ---
@st.cache_resource # Modelleri bir kez yükleyip önbelleğe almak için
def load_resources():
    # Ana model dosyasının URL'si (Google Drive'dan)
    # Bu model 'stacking_regressor_model.joblib' olacak ve URL'den indirilecek.
    MODEL_URL = "https://drive.google.com/uc?export=download&id=1RPnXBEpexRFLViV6orQL28yuo8XossVS" 

    # Modelin yerel olarak kaydedileceği dosya adı
    MODEL_PATH = "stacking_regressor_model.joblib" 

    # Diğer yardımcı joblib dosyalarının yolları (yerel olarak bulunacaklar)
    SCALER_PATH = "scaler.joblib" # Bu dosya adı düzeltildi
    ORIGINAL_X_COLUMNS_PATH = "original_X_columns.joblib"
    ALL_DESCRIPTIONS_PATH = "all_descriptions.joblib"
    NUMERICAL_FEATURES_PATH = "numerical_features.joblib"

    # Görsel dosyalarının yolları (yerel olarak bulunacaklar)
    required_images = [
        'sicaklik_nem_dagilimi.png',
        'sicaklik_nem_dagilimi_scatter.png',
        'santral.jpg' 
    ]

    # Büyük model dosyasını (stacking_regressor_model.joblib) URL'den indirme
    if not os.path.exists(MODEL_PATH):
        st.info(f"Büyük model dosyası '{MODEL_PATH}' indiriliyor, lütfen bekleyiniz...")
        try:
            # requests.Session kullanarak çerezleri ve oturum durumunu koru
            session = requests.Session()
            
            # İlk indirme denemesi (genellikle onay sayfasına veya doğrudan indirmeye yönlendirir)
            response = session.get(MODEL_URL, stream=True)
            response.raise_for_status() # HTTP hatalarını kontrol et (örn. 404 Not Found)

            # Google Drive'ın büyük dosyalar için onay sayfasını kontrol et
            # Genellikle bu sayfa HTML döner ve 'confirm' parametresi içerir.
            # Ayrıca yanıt başlıklarında 'Content-Disposition' olup olmadığını kontrol ederek doğrudan bir dosya mı yoksa HTML mi geldiğini anlarız.
            if 'Content-Type' in response.headers and 'text/html' in response.headers['Content-Type'] and 'Content-Disposition' not in response.headers:
                st.warning("Google Drive büyük dosya uyarısı algılandı. Onay sonrası tekrar indirme deneniyor...")
                
                # Onay sayfasından 'confirm' parametresini ayıkla
                # Bu regex deseni, Google Drive'ın onay sayfasındaki 'confirm' değerini bulmak için kullanılır.
                match = re.search(r'name="confirm" value="([A-Za-z0-9_]+)"', response.text)
                if match:
                    confirm_value = match.group(1)
                    # Yeni indirme URL'sini 'confirm' parametresiyle oluştur.
                    # Bazen bu adım için POST isteği gerekir.
                    confirmed_url = MODEL_URL + "&confirm=" + confirm_value
                    
                    # POST isteği ile onay ver ve gerçek indirmeyi başlat
                    response = session.post(confirmed_url, stream=True)
                    response.raise_for_status() # İkinci denemede de HTTP hatalarını kontrol et
                    st.info("Onay sonrası indirme isteği gönderildi.")
                else:
                    st.warning("Google Drive onay sayfası algılandı ancak 'confirm' parametresi bulunamadı. Dosya yine de indirilmeye çalışılıyor.")
            
            # Dosyayı kaydet
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Büyük model dosyası '{MODEL_PATH}' başarıyla indirildi.")
            # İndirilen dosyanın boyutunu kontrol et (hata ayıklama için)
            file_size_bytes = os.path.getsize(MODEL_PATH)
            st.info(f"İndirilen dosya boyutu: {file_size_bytes / (1024*1024):.2f} MB")

        except requests.exceptions.RequestException as e:
            st.error(f"""
                **HATA: Model dosyasını indirirken bir sorun oluştu!**
                **Detay:** {e}
                Lütfen '{MODEL_PATH}' dosyasının barındığı URL'nin doğru, erişilebilir ve herkese açık olduğundan emin olun.
                Google Drive'ın indirme sınırlamaları nedeniyle bu sorunlar yaşanabilir.
                
                **Traceback:**
                ```
                {traceback.format_exc()}
                ```
            """)
            st.stop()
    
    # Diğer joblib dosyaları (scaler, original_X_columns, all_descriptions, numerical_features)
    # yerel olarak yüklenir.
    required_joblibs_local = [
        MODEL_PATH, # Bu dosya artık yerel olarak var veya indirildi
        SCALER_PATH,
        ORIGINAL_X_COLUMNS_PATH,
        ALL_DESCRIPTIONS_PATH,
        NUMERICAL_FEATURES_PATH
    ]
    
    downloaded_objects = {}
    try:
        with st.spinner("Yardımcı dosyalar yerelden yükleniyor..."):
            for filename in required_joblibs_local:
                # Dosyanın yerel olarak var olup olmadığını kontrol et
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"'{filename}' dosyası bulunamadı. Lütfen projenizin ana dizininde (GitHub reposunda) olduğundan emin olun.")
                
                # Model dosyası zaten yukarıda işlendi veya indirildi, tekrar bilgi basmaya gerek yok
                if filename != MODEL_PATH:
                    st.info(f"'{filename}' yerelden yükleniyor (joblib.load)...")
                
                downloaded_objects[filename] = joblib.load(filename)
                
                if filename != MODEL_PATH:
                    st.success(f"'{filename}' başarıyla yüklendi!")

        # Yüklenen objeleri değişkenlere ata
        # 'lr_model' artık 'stacking_regressor_model.joblib' dosyasından yükleniyor
        lr_model = downloaded_objects[MODEL_PATH] 
        scaler = downloaded_objects[SCALER_PATH]
        original_X_columns = downloaded_objects[ORIGINAL_X_COLUMNS_PATH]
        all_descriptions = downloaded_objects[ALL_DESCRIPTIONS_PATH]
        numerical_features = downloaded_objects[NUMERICAL_FEATURES_PATH]

        # Görsel dosyalarının varlığını kontrol et (sadece bilgilendirme)
        for img in required_images:
            if not os.path.exists(img):
                st.warning(f"Görsel '{img}' bulunamadı. Lütfen model eğitim dosyasını (energy_prediction_model.ipynb) çalıştırdığınızdan ve görsellerin aynı dizine kaydedildiğinden emin olun.")
                
        return lr_model, scaler, original_X_columns, all_descriptions, numerical_features
    
    except FileNotFoundError as e:
        st.error(f"""
            **HATA: Gerekli dosyalardan biri bulunamadı!**
            **Detay:** {e}
            Lütfen projenizin tüm model, yardımcı ve görsel dosyalarının Streamlit uygulamanızla **aynı dizinde** (GitHub reposunda) olduğundan emin olun.
            Bu dosyaları oluşturmak için lütfen **model eğitim dosyasını (energy_prediction_model.ipynb) çalıştırın**.
            
            **Traceback:**
            ```
            {traceback.format_exc()}
            ```
        """)
        st.stop()
    except Exception as e: # joblib yükleme sırasındaki PicklingError veya diğer bilinmeyen hataları yakalamak için
        st.error(f"""
            **HATA: Model/Yardımcı dosyalardan biri yüklenirken beklenmeyen bir sorun oluştu!**
            Bu genellikle, modelin kaydedildiği ortam ile yüklendiği ortam arasındaki kütüphane sürümü uyumsuzluklarından kaynaklanır.
            **Detay:** {e}
            
            **Çözüm Önerisi:** Lütfen yerel ortamınızdaki tüm kütüphanelerin (özellikle `joblib`, `scikit-learn`, `numpy`, `pandas`, `loky`) modelin kaydedildiği sürümle tam olarak eşleştiğinden emin olun. Gerekirse Conda ortamınızı silip `requirements.txt` ile yeniden kurun ve modelleri yeniden kaydedin (`protocol=4` ile).
            
            **Traceback:**
            ```
            {traceback.format_exc()}
            ```
        """)
        st.stop()


# Dosyaları yükle
lr_model, scaler, original_X_columns, all_descriptions, numerical_features = load_resources()


# --- Sunum Kısmı ---

st.title("Enerji Tüketimi Tahmin Uygulaması Sunumu")
st.markdown("---") # Ayırıcı

# Slayt 1: Giriş ve Problem Tanımı
st.header("1. Giriş ve Problem Tanımı")
st.write("""
    Günümüz dünyasında, enerji kaynaklarının verimli kullanımı ve sürdürülebilir enerji yönetimi, çevresel, ekonomik ve sosyal açıdan büyük bir öneme sahiptir. Enerji tüketiminin doğru bir şekilde tahmin edilmesi, enerji üretim planlamasından akıllı şebeke yönetimine, maliyet optimizasyonundan karbon emisyonlarının azaltılmasına kadar birçok alanda kritik faydalar sunar. Geleneksel yöntemler genellikle statik ve sınırlı kalırken, makine öğrenimi modelleri dinamik ve karmaşık ilişkileri öğrenerek daha doğru tahminler yapma potansiyeli sunar.
    
    Bu projemizde, enerji tüketiminin temel sürücülerini anlamak ve gelecekteki aktif güç (kW cinsinden) tüketimini yüksek doğrulukla tahmin etmek amacıyla bir makine öğrenimi modeli geliştirilmesi hedeflenmiştir. Elde edilen modelin, kullanıcı dostu bir web uygulaması (Streamlit) aracılığıyla erişilebilir kılınması, teorik bilginin pratik bir araca dönüştürülmesini sağlamaktadır.
""")
st.markdown("""
* **Projenin Amacı:** Çeşitli hava durumu verileri (sıcaklık, basınç, nem, rüzgar hızı ve yönü) ve elektrik şebekesi parametrelerini (akım, voltaj) kullanarak aktif güç tüketimini doğru bir şekilde tahmin eden robust bir makine öğrenimi modeli geliştirmek ve bu modeli interaktif bir Streamlit uygulaması aracılığıyla son kullanıcılara sunmaktır.
* **Veri Seti:** Projemizin temelini, enerji tüketimi (aktif güç) verileri ile zenginleştirilmiş, eş zamanlı hava durumu verilerini içeren kapsamlı bir veri seti oluşturmaktadır. Bu veri seti, modelin karmaşık çevresel ve elektriksel etkenler arasındaki ilişkileri öğrenmesi için zemin hazırlamıştır.
""")

# Santral görselini ekleme: Eğer santral.jpg bulunamazsa placeholder kullan
if os.path.exists('santral.jpg'):
    st.image("santral.jpg", caption="Enerji Santrali Örneği", width=1250)
else:
    st.image("https://placehold.co/800x450/333333/FFFFFF?text=Enerji%20Santrali%20Görseli", caption="Enerji Santrali Örneği (Görsel Bulunamadı)", use_column_width=True)

st.markdown("---")

# Slayt 2: Veri Analizi ve Ön İşleme
st.header("2. Veri Analizi ve Ön İşleme")
st.write("""
    Veri bilimi projelerinin temelini oluşturan veri analizi ve ön işleme aşaması, ham verinin kullanılabilir ve model için optimize edilmiş bir formata dönüştürülmesini içerir. Bu aşama, modelin performansını doğrudan etkileyen kritik bir adımdır.
""")
st.subheader('2.1. Veri Seti Keşfi ve Temizliği')
st.write("""
* **Veri Yükleme ve Genel Bakış:** Projenin başlangıcında, `energy_weather_raw_data.csv` adlı ham veri seti `pandas` kütüphanesi kullanılarak yüklenmiştir. Veri setinin sütun yapıları (`df.columns`), ilk beş satırı (`df.head()`), istatistiksel özetleri (`df.describe()`) ve veri tipleri (`df.info()`) detaylıca incelenmiştir.
* **Eksik Değer Analizi:** Veri setinde herhangi bir eksik (NaN) değer olup olmadığı kontrol edilmiş ve tüm sütunların tam olduğu, dolayısıyla eksik değer doldurma (imputation) ihtiyacının olmadığı tespit edilmiştir. Bu durum, veri setinin kalitesi açısından olumlu bir göstergedir.
* **Gereksiz Sütunların Atılması:** `date` sütunu, doğrudan tahminlemeye katkıda bulunmadığı ve daha karmaşık zaman serisi analizleri gerektireceği için modelden çıkarılmıştır. Bu, modelin odağını belirlenen fiziksel ve çevresel özelliklere kaydırmıştır.
""")
st.subheader('2.2. Aykırı Değer Analizi')
st.write("""
* **IQR Metodu Uygulaması:** Sayısal sütunlardaki aykırı değerlerin (outliers) tespiti için Çeyrekler Arası Aralık (IQR - Interquartile Range) metodu kullanılmıştır. Bu metot, verinin dağılımına dayanarak alt ve üst sınırları belirler (Q1 - 1.5*IQR ve Q3 + 1.5*IQR). Bu sınırların dışında kalan değerler potansiyel aykırı değer olarak kabul edilir.
* **Tespit ve Yönetim:** Analiz sonucunda, belirli sütunlarda (örn: `active_power`, `current`, `temp`) aykırı değerler tespit edilmiştir. Ancak bu değerlerin sistemsel hatalardan ziyade, anlık yüksek yüklenmeler veya anormal hava koşulları gibi gerçek senaryoları yansıtabileceği değerlendirilerek modelin genellenebilirliğini artırmak amacıyla direkt olarak çıkarılmamıştır. Bu yaklaşım, modelin daha robust olmasını hedefler.
""")
st.subheader('2.3. Özellik Mühendisliği ve Dönüşümü')
st.write("""
* **Kategorik Veri Dönüşümü (One-Hot Encoding):** `description` (hava durumu açıklaması) gibi kategorik sütunlar, makine öğrenimi modellerinin anlayabileceği sayısal formata dönüştürülmüştür. Bu dönüşüm için `pd.get_dummies` kullanılarak One-Hot Encoding yöntemi tercih edilmiştir. Her benzersiz kategori için ayrı bir ikili (0/1) sütun oluşturulmuştur (örn: `description_clear_sky`, `description_broken_clouds`). Bu sayede model, farklı hava durumu açıklamalarının aktif güç üzerindeki etkisini ayrı ayrı öğrenebilir.
* **Özellik Ölçeklendirme (StandardScaler):** Model performansını optimize etmek ve gradient tabanlı algoritmaların daha hızlı ve doğru bir şekilde yakınsamasını sağlamak amacıyla **yalnızca sayısal özellikler** (`current`, `voltage`, `temp`, `pressure`, `humidity`, `speed`, `deg`) `StandardScaler` ile ölçeklendirilmiştir. Bu işlem, her bir sayısal özelliği ortalaması 0 ve standart sapması 1 olacak şekilde dönüştürür. One-Hot Encoded sütunlar ikili yapılarından dolayı ölçeklendirme işlemine dahil edilmemiştir.
* **Öznitelik Bağıntı Analizi (VIF):** `statsmodels` kütüphanesi kullanılarak Varyans Büyütme Faktörü (VIF - Variance Inflation Factor) analizi yapılmıştır. Bu analiz, bağımsız değişkenler arasındaki çoklu doğrusal bağıntıyı (multicollinearity) tespit etmek için kullanılır. Yüksek VIF değerine sahip (`active_power`, `current`, `apparent_power`, `reactive_power`, `temp`, `feels_like`, `temp_t+1`, `feels_like_t+1`) bazı sütunlar tespit edilmiştir. `active_power` hedef değişkenimiz olduğu için `drop` edilmemiştir. `apparent_power` ve `reactive_power` ise `active_power` ile güçlü matematiksel ilişkisi olduğundan (güç formülü) modelin karmaşıklığını ve olası aşırı uyumu azaltmak için çıkarılmıştır. `temp_t+1` ve `feels_like_t+1` gibi geleceğe yönelik sıcaklık tahminleri de `temp` ve `feels_like` ile yüksek korelasyona sahip oldukları ve modelin mevcut zaman anındaki tahmini odaklandığı için çıkarılmıştır.

""")
st.subheader('2.4. Veri Görselleştirme ile İlişkileri Keşfetme')
st.write("""
    Veri setindeki temel ilişkileri anlamak ve modelin öğreneceği potansiyel kalıpları görsel olarak keşfetmek için çeşitli grafikler kullanılmıştır.
""")

st.markdown("##### Sıcaklık ve Nem Dağılımı")
if os.path.exists('sicaklik_nem_dagilimi.png'):
    st.image('sicaklik_nem_dagilimi.png', caption='Sıcaklık aralıklarına göre nem oranlarının kutu grafiği, medyan ve çeyrek değerleri gösterir.', width=1400)
else:
    st.warning("Görsel 'sicaklik_nem_dagilimi.png' bulunamadı.")

st.markdown("##### Sıcaklık-Nem İlişkisi")
if os.path.exists('sicaklik_nem_dagilimi_scatter.png'):
    st.image('sicaklik_nem_dagilimi_scatter.png', caption='Sıcaklık ve nem arasındaki genel ilişkiyi gösteren dağılım grafiği.', width=1400)
else:
    st.warning("Görsel 'sicaklik_nem_dagilimi_scatter.png' bulunamadı.")

st.subheader('Aylara Göre Nem İlişkisi')
try:
    st.image('ay_nem.jpeg', caption='Veri Setindeki Aylara Göre Nem İlişkisi', width=1400)
except FileNotFoundError:
    st.warning("Görsel 'ay_nem.jpeg' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")

st.subheader('Aylara Göre Güç İlişkisi')
try:
    st.image('ay_guc.jpeg', caption='Veri Setindeki Aylara Göre Güç İlişkisi (Kutu Grafiği)', width=1400)
except FileNotFoundError:
    st.warning("Görsel 'ay_guc.jpeg' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")
st.markdown("---")

st.subheader('Aylara Göre Sıcaklık İlişkisi')
try:
    st.image('ay_sıcaklık.jpeg', caption='Veri Setindeki Aylara Göre Sıcaklık İlişkisi (Kutu Grafiği)', width=1400)
except FileNotFoundError:
    st.warning("Görsel 'ay_sıcaklık.jpeg' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")

st.subheader('Hava Durumu ve Güç İlişkisi')
try:
    st.image('hava_guc.jpeg', caption='Veri Setindeki Hava Durumu ve Güç İlişkisi (Kutu Grafiği)', width=1400)
except FileNotFoundError:
    st.warning("Görsel 'hava_sıcaklık.jpeg' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")
st.markdown("---")

st.subheader('Saat ve Güç İlişkisi')
try:
    st.image('saat_guc.jpeg', caption='Veri Setindeki Saat ve Güç İlişkisi', width=1400)
except FileNotFoundError:
    st.warning("Görsel 'saat_guc.jpeg' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")
st.markdown("---")

# Slayt 3: Model Geliştirme Stratejisi
st.header('3. Stratejimiz: "Uzmanlar Komitesi" Yaklaşımı ile En İyi Modeli İnşa Etmek')
st.markdown("En doğru tahmini yapmak için tek bir 'sihirli' model aramak yerine, farklı modellerin güçlü yönlerini birleştiren bir strateji benimsedik. Bu, tek bir uzmana danışmak yerine, farklı alanlarda uzmanlaşmış bir uzmanlar komitesinden görüş almaya benzer.")
st.subheader("3.1. Aday Modeller ve Topluluk Öğrenmesinin Gücü")
st.markdown("""
- **Test Edilen Modeller:** Projede, farklı yeteneklere sahip model ailelerini karşılaştırdık:
    - **Doğrusal Modeller (Ridge, Lasso):** Hızlı, yorumlanabilir ve iyi bir başlangıç noktası sunan temel modeller.
    - **Ağaç Tabanlı Modeller (Random Forest, LightGBM, XGBoost):** Karmaşık ve doğrusal olmayan ilişkileri yakalamada son derece başarılı, modern ve güçlü algoritmalar.
- **Hiperparametre Optimizasyonu (`GridSearchCV`):** Her modelin potansiyelini en üst düzeye çıkarmak için, en iyi ayarları (örneğin bir ormandaki ağaç sayısı, öğrenme oranı vb.) sistematik olarak bulan `GridSearchCV` tekniğini kullandık.
- **Stacking Regressor (Nihai Yaklaşımımız):** Bu, sıradan bir oylamadan daha fazlasıdır. Bu, hiyerarşik bir uzmanlık sistemidir:
    1.  **1. Kademe (Uzmanlar):** En güçlü modellerimiz (Random Forest, LGBM, XGBoost), veriyi analiz eder ve kendi tahminlerini üretir. Her biri probleme farklı bir açıdan bakar.
    2.  **2. Kademe (Yönetici Meta-Model):** Daha sonra, `Ridge` adında bir "yönetici" model devreye girer. Bu modelin tek işi, uzmanların tahminlerini incelemek ve hangi uzmanın hangi koşullar altında daha güvenilir olduğunu öğrenmektir. Sonuçta, bu uzman görüşlerini akıllıca birleştirerek nihai ve daha isabetli bir karar verir.
""")
st.info("**Projemizin nihai modeli, bireysel uzmanların bilgeliğini birleştiren bu gelişmiş Stacking mimarisidir.**")
st.markdown("---")

# Slayt 4: Model Performansı ve Sonuçların Doğrulanması
st.header("4. Sonuçların Analizi: Modelimiz Sadece Tahmin mi Ediyor, Yoksa Anlıyor mu?")
st.markdown("Bir model geliştirmek denklemin bir yarısıdır. Diğer yarısı ise bu modelin ne kadar güvenilir olduğunu kanıtlamaktır. Modelimizin sadece geçmişi ezberlemediğinden, geleceği öngörebilecek şekilde gerçekten 'öğrendiğinden' emin olmalıydık.")
st.subheader("4.1. Başarı Metrikleri: R²'nin Ötesinde")
st.markdown("""
- **R² Skoru (Belirlilik Katsayısı):** Nihai modelimiz, test verisi üzerinde **R² = 0.98** gibi olağanüstü bir skora ulaştı. Bunun anlamı şudur: Enerji tüketimindeki dalgalanmaların **%98'inin nedenini** modelimizdeki faktörlerle açıklayabiliyoruz. Geriye kalan %2'lik kısım, verimizde bulunmayan öngörülemez insan davranışları veya ani olaylar gibi faktörlere aittir.
- **Model Karşılaştırması:** Stacking yaklaşımının gücünü göstermek için, tekil modellerin performansıyla karşılaştırdık. Sonuçlar, "uzmanlar komitesinin" tek bir uzmandan her zaman daha bilge olduğunu kanıtladı.
""")

st.subheader("4.2. Güvenilirlik Testi: Aşırı Uyum (Overfitting) ve Çapraz Doğrulama")
st.markdown("""
İyi bir model sadece bildiği soruları değil, daha önce hiç görmediği soruları da cevaplayabilmelidir.
- **Eğitim vs. Test Skorları:** Modelimizin eğitim verisindeki performansı ile daha önce hiç görmediği test verisindeki performansının neredeyse aynı olması (R² ≈ 0.98), modelimizin ezber yapmadığının en güçlü kanıtıdır.
- **K-Katlı Çapraz Doğrulama:** En titiz testimiz buydu. Veri setini 5 parçaya ayırdık ve modelimizi 5 kez, her seferinde farklı bir parçayı test verisi olarak kullanarak eğittik. Sonuçların (örn: R² ortalaması ≈ 0.981, standart sapma ≈ 0.002) son derece tutarlı olması, model performansının şansa bağlı olmadığını ve sağlam temellere dayandığını bize gösterdi. Modelimiz bir **tarihçi değil, güvenilir bir kahindir.**
""")
st.markdown("---")

st.header("5. Projenin Etkisi ve Uygulama Alanları")
st.markdown("""
Yüksek doğruluklu bir model geliştirmek, teknik bir başarıdır. Ancak projenin asıl değeri, bu teknik başarının gerçek dünyada yarattığı **somut faydalarda** yatar. Bu model, farklı paydaşlar için stratejik bir karar destek aracı olarak hizmet edebilir:
""")
st.markdown("""
- **Akıllı Şebeke Yönetimi:**
  - **Yük Dengeleme:** Tahminler, operatörlerin enerji yükünü şebeke genelinde proaktif olarak dengelemesine olanak tanır. Bu, ekipman ömrünü uzatır ve teknik arızaları azaltır.
  - **Kesinti Önleme:** Beklenen aşırı yüklenmeler önceden tespit edilerek, planlı bakım ve kapasite artırımı gibi önlemler alınabilir, böylece beklenmedik kesintilerin önüne geçilir.

- **Enerji Ticareti ve Piyasalar:**
  - **Kârlı Alım-Satım:** Enerji talebinin ne zaman artıp azalacağını öngörmek, enerji şirketlerinin spot piyasalarda daha kârlı alım-satım işlemleri yapmasını sağlar. Düşük talep zamanlarında ucuza alıp, yüksek talep zamanlarında pahalıya satabilirler.
  - **Risk Yönetimi:** Fiyat dalgalanmalarına karşı daha hazırlıklı olmayı ve finansal riskleri minimize etmeyi sağlar.

- **Yenilenebilir Enerji Entegrasyonu:**
  - **Volatilitenin Yönetimi:** Model, güneş veya rüzgar üretiminin düşeceği zamanlarda konvansiyonel santrallerin ne kadar devreye girmesi gerektiğini tahmin ederek, yenilenebilir kaynakların şebekeye sorunsuz entegrasyonunu kolaylaştırır.

- **Tesis ve Tüketici Yönetimi:**
  - **Verimlilik Artışı:** Büyük endüstriyel tesisler veya ticari binalar, kendi tüketimlerini tahmin ederek enerji maliyetlerini optimize edebilir ve üretim planlarını en verimli şekilde yapabilirler.
""")
st.markdown("---")

# Slayt 6: Enerji Tahmin Uygulaması
st.header("6. Canlı Enerji Tahmin Uygulaması")
st.write("""
    Geliştirdiğimiz bu interaktif web uygulaması, modelimizin pratik kullanımını ve tahmin yeteneğini göstermektedir.
    Aşağıdaki bölümde, istediğiniz parametreleri girerek aktif güç tüketimi için anında tahminler alabilirsiniz.
""")
st.markdown("---")

# --- Uygulama Kısmı ---
st.subheader('Aktif Güç Tahmini Yapın')

col1, col2, col3 = st.columns(3)

with col1:
    current = st.number_input('Akım (Current)', min_value=0.0, value=2.53, format="%.2f") # Default values set to first row
    voltage = st.number_input('Voltaj (Voltage)', min_value=0.0, value=122.20, format="%.2f")
    temp = st.number_input('Sıcaklık (°C)', min_value=-50.0, value=24.19, format="%.2f")

with col2:
    pressure = st.number_input('Basınç (hPa)', min_value=0.0, value=1013.00, format="%.2f")
    humidity = st.number_input('Nem (%)', min_value=0.0, max_value=100.0, value=39.00, format="%.2f")
    speed = st.number_input('Rüzgar Hızı (m/s)', min_value=0.0, value=0.00, format="%.2f")

with col3:
    deg = st.number_input('Rüzgar Yönü (°)', min_value=0.0, max_value=360.0, value=0.00, format="%.2f")
    
    # Get all unique description values (loaded from saved file)
    # Set default value to 'clear sky'
    default_description_index = all_descriptions.index('clear sky') if 'clear sky' in all_descriptions else 0
    description = st.selectbox('Hava Durumu Açıklaması (Description)', all_descriptions, index=default_description_index)


# Predict button
if st.button('Aktif Güç Tahmin Et'):
    # Convert user inputs to a DataFrame
    input_data = pd.DataFrame([[current, voltage, temp, pressure, humidity, speed, deg, description]],
                               columns=['current', 'voltage', 'temp', 'pressure', 'humidity', 'speed', 'deg', 'description'])

    # Apply one-hot encoding to the input
    input_encoded = pd.get_dummies(input_data, columns=['description'], dtype='int')

    # Ensure all columns from original_X_columns are present (fill missing with 0)
    final_input = input_encoded.reindex(columns=original_X_columns, fill_value=0)

    # Debug output: Show the final DataFrame prepared for prediction

    # Scale only the numerical features
    final_input[numerical_features] = scaler.transform(final_input[numerical_features])

    # Debug output: Show the scaled DataFrame before entering the model

    # Make prediction (using the loaded Stacking Regressor model)
    prediction = lr_model.predict(final_input)[0]

    st.subheader('Tahmin Edilen Aktif Güç:')
    st.success(f'{prediction:.2f} kW')

st.markdown("---")

st.subheader("Dinlediğiniz için teşekkürler!")
st.write("Projemizi incelediğiniz için teşekkür ederiz. Sorularınız varsa memnuniyetle cevaplayabiliriz.")
