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

# 4. slayt
st.header("4. Sonuçların Analizi: Modelimiz Tahmin Etmekle mi Yetiniyor, Yoksa Gerçekten Anlıyor mu?")
st.markdown("""
Bir makine öğrenmesi modeli geliştirmek sadece verileri işleyip doğru tahminler almakla sınırlı değildir. Asıl hedef, modelin bu tahminleri nasıl yaptığına dair bir anlayış geliştirmek ve modelin sadece geçmişi ezberleyip ezberlemediğini değil, geleceği de güvenilir şekilde öngörebilecek kadar "anlayıp anlamadığını" test etmektir. Enerji tüketimi gibi çok sayıda faktörün etkilediği bir konuda, modelimizin gerçekten genellenebilir ve sağlam bir yapıya sahip olması kritik öneme sahip. Bu bölümde, modelimizin bu yetkinliğe ulaşıp ulaşmadığını hangi metriklerle ve yöntemlerle incelediğimizi anlatıyoruz.
""")

st.subheader("4.1. Başarı Metrikleri: Modelin Gerçekten Öğrenip Öğrenmediğini Anlamak")

st.markdown("""
* **R² Skoru – Model Ne Kadar Açıklayabiliyor?:**  
    Nihai stacking modelimiz, test verisi üzerinde **R² = 0.98** skoruna ulaştı. Bu, enerji tüketimindeki değişimlerin %98’inin modelde yer alan faktörler (hava durumu, zaman, ekonomi vb.) tarafından açıklanabildiğini gösteriyor. Kalan %2 ise çoğunlukla öngörülemeyen olaylar veya dışsal etkenlerden kaynaklanıyor. Bu kadar yüksek bir skor, modelimizin enerji tüketimini belirleyen temel etkenleri oldukça doğru yakaladığını gösteriyor.
""")

st.markdown("""
* **Model Karşılaştırması – Birlikten Kuvvet Doğar:**  
    Modelimizi oluşturan tekil modellerle (örneğin Random Forest, XGBoost, LightGBM) stacking modelini karşılaştırdık. Beklendiği gibi, stacking yaklaşımı her bir modelin güçlü yönlerinden faydalanarak daha yüksek doğruluk sağladı. Tekil modeller belli veri yapılarında iyi performans gösterse de, stacking modeli genel başarıyı artırarak daha dengeli ve güvenilir sonuçlar verdi.
""")

st.subheader("4.2. Güvenilirlik Testleri: Aşırı Uyum ve Çapraz Doğrulama")
st.markdown("""
Modelin sadece geçmiş verilerde değil, daha önce hiç görmediği verilerde de iyi performans göstermesi gerekiyor. Bu yüzden modelimizin genelleyici olup olmadığını test etmek için farklı güvenilirlik analizleri yaptık.
""")

st.markdown("""
* **Eğitim vs. Test Performansı – Ezberleyen mi, Öğrenen mi?:**  
    Eğitim ve test setlerinde elde edilen R² skorlarının birbirine çok yakın (her ikisi de yaklaşık **0.98**) olması, modelin ezber yapmadığını ve genelleme yeteneğinin yüksek olduğunu gösteriyor. Bu, modelin yalnızca veriye değil, verinin taşıdığı anlam ve örüntülere hakim olduğunu kanıtlıyor.
""")

st.markdown("""
* **K-Katlı Çapraz Doğrulama – Gerçek Dayanıklılık Testi:**  
    Modeli farklı veri bölümleriyle test etmek için 5 katlı çapraz doğrulama uyguladık. Her bir katmanda eğitim ve test işlemi tekrarlanarak modelin tutarlılığı ölçüldü. Sonuçlar oldukça etkileyiciydi: Ortalama R² ≈ 0.981, standart sapma ise ≈ 0.002. Bu kadar düşük bir sapma, modelin her veri grubunda benzer performans gösterdiğini, yani sağlam ve güvenilir olduğunu gösteriyor. Başka bir deyişle, modelimiz sadece geçmişi anlatan bir araç değil; geleceği tahmin edebilen güçlü bir sistem.
""")

st.markdown("""
Bu bölümdeki kapsamlı analizler ve güvenilirlik testleri, modelimizin sadece yüksek performanslı tahminler yapmakla kalmayıp, aynı zamanda enerji tüketimi dinamiklerini derinlemesine anladığını ve bu sayede gerçek dünya problemlerine uygulanabilir sağlam içgörüler sunduğunu kanıtlamıştır.
""")

st.markdown("---")


st.header("5. Projenin Etkisi ve Uygulama Alanları")
st.markdown("""
Yüksek doğruluklu bir enerji tüketimi tahmin modeli geliştirmek, kesinlikle önemli bir teknik başarıdır. Ancak bu projenin gerçek değerini oluşturan, bu başarının enerji sektöründeki farklı alanlarda yarattığı **somut faydalar** ve açtığı yeni uygulama fırsatlarıdır. Geliştirdiğimiz bu model, pek çok farklı paydaş için stratejik bir karar destek aracı olarak kullanılabilir. İşte bu potansiyelin detayları:
""")

st.subheader("5.1. Akıllı Şebeke Yönetimi – Şebekeyi Daha Akıllı ve Dayanıklı Hale Getirmek")
st.markdown("""
Enerji şebekelerinin karmaşıklığı göz önüne alındığında, doğru ve zamanında yapılan tüketim tahminleri, şebeke operatörleri için hayati önem taşır.
* **Yük Dengeleme ve Optimizasyon:**
    * Modelimiz, enerji yükünü proaktif bir şekilde dengeleme konusunda operatörlere yardımcı olur. Elektrik yükünün hangi bölgelerde ne zaman artıp azalacağına dair tahminler sayesinde, santrallerin ve trafoların çıktıları buna göre optimize edilebilir. Bu, aşırı yüklenmelerin ve yetersiz beslemelerin önüne geçilmesini sağlar.
    * Bu tür optimizasyon, aynı zamanda şebeke ekipmanlarının ömrünü uzatır ve teknik arızaların sayısını önemli ölçüde azaltır. Bu da işletme ve bakım maliyetlerinde belirgin bir düşüş sağlar.
* **Kesinti Önleme ve Güvenilirlik Artışı:**
    * Model, beklenen aşırı yüklenmeleri veya talep artışlarını önceden tespit edebilir. Bu erken uyarı sistemi, enerji şirketlerinin planlı bakım, kapasite artırımı veya alternatif enerji kaynakları devreye alma süreçlerini daha etkin bir şekilde yönetmelerini sağlar.
    * Proaktif müdahaleler sayesinde, beklenmedik ve geniş çaplı elektrik kesintilerinin önüne geçilerek, şebeke güvenilirliği ve enerji arz güvenliği artırılır.
""")

st.subheader("5.2. Enerji Ticareti ve Piyasalar – Kârı Artırmak ve Riski Azaltmak")
st.markdown("""
Enerji piyasaları, dinamik fiyat dalgalanmaları ve anlık arz-talep dengesizlikleri ile şekillenir. Doğru tahminler, bu piyasada rekabet avantajı elde etmeye yardımcı olur.
* **Kârlı Alım-Satım Stratejileri:**
    * Enerji talebinin hangi saatlerde, günlerde veya mevsimsel olarak artıp azalacağına dair doğru tahminler, enerji şirketlerinin spot piyasalarda daha kârlı alım-satım işlemleri yapabilmelerini sağlar. Örneğin, talep düşükken (fiyatlar uygun olduğunda) enerji satın alıp depolayabilirler ve talep yüksekken (fiyatlar arttığında) bu enerjiyi satabilirler.
    * Bu strateji, şirketlerin gelirlerini artırırken maliyetlerini de düşürmelerine olanak tanır.
* **Finansal Risk Yönetimi:**
    * Enerji fiyatları, arz ve talep dengesine göre hızla değişebilir. Modelimiz, bu dalgalanmaları öngörerek enerji şirketlerinin fiyat değişimlerine karşı daha hazırlıklı olmalarını sağlar.
    * Bu da belirsizliği azaltarak finansal riskleri minimize eder ve daha sağlam bütçe planlamalarına olanak tanır, yatırımcı güvenini artırır.
""")

st.subheader("5.3. Yenilenebilir Enerji Entegrasyonu – Yeşil Enerjinin Şebekeye Sorunsuz Katılımı")
st.markdown("""
Yenilenebilir enerji kaynakları (güneş, rüzgar) çevresel faydaları yüksek olsa da, değişken ve öngörülemez doğaları nedeniyle şebekeye entegrasyonları zordur.
* **Volatilitenin Akıllı Yönetimi:**
    * Modelimiz, geleneksel enerji kaynaklarından bağımsız olarak enerji talebini doğru şekilde tahmin edebildiğinden, güneş veya rüzgar enerjisi üretiminin azaldığı zamanlarda (örneğin, bulutlu günler veya rüzgarsız havalar) konvansiyonel santrallerin üretmesi gereken ek enerji miktarını belirleyebilir.
    * Bu senkronizasyon, yenilenebilir enerji kaynaklarının şebekeye sorunsuz bir şekilde entegrasyonunu sağlar, fazla üretimi engellerken talep karşılamada eksiklik yaşanmasını önler.
* **Hibrit Sistem Optimizasyonu:**
    * Model, yenilenebilir kaynakların değişkenliğini göz önünde bulundurarak, hibrit enerji sistemlerinde (örneğin güneş panelleri ve batarya depolama sistemleri) batarya şarj/deşarj stratejilerini optimize edebilir. Bu, yenilenebilir enerjiden elde edilen faydayı maksimum düzeye çıkarır.
""")

st.subheader("5.4. Tesis ve Tüketici Yönetimi – Verimlilik ve Maliyet Tasarrufu")
st.markdown("""
Yalnızca büyük enerji şirketleri değil, bireysel tüketiciler ve endüstriyel tesisler de bu modelin sunduğu avantajlardan faydalanabilir.
* **Büyük Tesisler İçin Verimlilik Artışı:**
    * Endüstriyel tesisler, ticari binalar ve kampüsler, modelin tahminlerinden yararlanarak enerji tüketimlerini optimize edebilirler. Enerji yoğun süreçlerini, elektrik fiyatlarının daha düşük olduğu saatlerde gerçekleştirerek maliyetlerini önemli ölçüde düşürebilirler.
    * Bu strateji, üretim süreçlerini enerji maliyetlerine göre şekillendirme yeteneği sunarak operasyonel verimlilik ve rekabet gücünü artırır.
* **Tüketici Bilinçlendirmesi ve Talep Yanıtı:**
    * Modelin sağladığı tahminler, akıllı ev sistemleri veya tüketici arayüzleri aracılığıyla bireysel tüketicilere sunulabilir. Bu sayede, tüketiciler daha bilinçli bir şekilde enerji tüketimlerini yönetebilirler. Örneğin, elektrik fiyatlarının artacağı saatler önceden bildirildiğinde, enerji yoğun cihazlar (örneğin, çamaşır makineleri) daha uygun saatlerde çalıştırılabilir.
    * Bu, toplam enerji talebinin yönetilmesine yardımcı olur ve şebeke üzerindeki yükün azaltılmasına katkı sağlar.
""")

st.markdown("""
Sonuç olarak, geliştirdiğimiz bu enerji tüketimi tahmin modeli sadece gelişmiş bir yapay zeka algoritması değil, aynı zamanda enerji sektöründeki karar vericiler için güçlü bir araçtır. Yüksek doğruluğu ve sağladığı içgörüler sayesinde, daha sürdürülebilir, verimli ve güvenilir bir enerji geleceğine ulaşmak için önemli bir adım atılmasını sağlar.
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
