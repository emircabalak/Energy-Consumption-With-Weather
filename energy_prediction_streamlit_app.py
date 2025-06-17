import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests # Dosyaları indirmek için

st.set_page_config(layout="wide")

# --- Gerekli Dosyaların Yüklendiğinden Emin Olma Fonksiyonu ---
@st.cache_resource # Modelleri bir kez yükleyip önbelleğe almak için
def load_resources():
    # *** SADECE ANA MODELİN GOOGLE DRIVE URL'SİNİ BURAYA YAPIŞTIRIN ***
    MODEL_URL = "https://drive.google.com/file/d/1MXmLHu3TS04B7psZ72zBA7Tv3zeUYFK_/view?usp=sharing"

    # Diğer .joblib dosyaları ve görsellerin yerel (GitHub reposunda) olduğu varsayılıyor.
    required_local_joblibs = [
        'scaler.joblib',
        'original_X_columns.joblib',
        'all_descriptions.joblib',
        'numerical_features.joblib'
    ]
    required_local_images = [
        'sicaklik_nem_dagilimi.png',
        'sicaklik_nem_dagilimi_scatter.png',
        'santral.png'
    ]

    try:
        # Ana modeli Google Drive'dan indir
        with st.spinner("Ana model yükleniyor..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # HTTP hatalarını kontrol et
            with open('stacking_regressor_model.joblib', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            lr_model = joblib.load('stacking_regressor_model.joblib')

        # Diğer .joblib dosyalarını yerel olarak yükle
        with st.spinner("Yardımcı dosyalar yükleniyor..."):
            scaler = joblib.load('scaler.joblib')
            original_X_columns = joblib.load('original_X_columns.joblib')
            all_descriptions = joblib.load('all_descriptions.joblib')
            numerical_features = joblib.load('numerical_features.joblib')

        st.success("Tüm model ve yardımcı dosyalar başarıyla yüklendi!")
        
        # Görsel dosyalarının varlığını kontrol et (sadece bilgilendirme)
        for img in required_local_images:
            if not os.path.exists(img):
                st.warning(f"Görsel '{img}' bulunamadı. Lütfen model eğitim dosyasını (energy_prediction_model.py) çalıştırdığınızdan ve görsellerin aynı dizine kaydedildiğinden emin olun.")
            
        return lr_model, scaler, original_X_columns, all_descriptions, numerical_features
    
    except FileNotFoundError as e:
        st.error(f"""
            **Gerekli yerel model veya yardımcı dosyalar bulunamadı!**
            Detay: {e}
            Lütfen aşağıdaki dosyaların Streamlit uygulamanızla aynı dizinde (ve GitHub deponuzda) olduğundan emin olun:
            - `scaler.joblib`
            - `original_X_columns.joblib`
            - `all_descriptions.joblib`
            - `numerical_features.joblib`
            - `sicaklik_nem_dagilimi.png`
            - `sicaklik_nem_dagilimi_scatter.png`
            - `santral.png`
            
            Bu dosyaları oluşturmak için lütfen **Canvas belgesini (energy_prediction_model) çalıştırın**.
            Ardından bu Streamlit uygulamasını tekrar başlatın.
        """)
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"""
            **Ana model Google Drive'dan indirilirken bir hata oluştu!**
            Detay: {e}
            Lütfen Google Drive URL'nizin doğru olduğundan ve dosyanıza herkese açık erişimin verildiğinden emin olun.
        """)
        st.stop()
    except Exception as e:
        st.error(f"""
            **Bir hata oluştu!**
            Detay: {e}
            Lütfen tüm dosyaların doğru şekilde yapılandırıldığından emin olun.
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
    Enerji sektöründe verimlilik ve maliyet yönetimi, günümüzde büyük önem taşımaktadır. 
    Kaynakların doğru planlanması ve şebeke istikrarının sağlanması, aynı zamanda sürdürülebilirlik hedeflerine ulaşma açısından kritik rol oynamaktadır.
""")
st.markdown("""
* **Projenin Amacı:** Hava durumu ve elektrik parametrelerini kullanarak aktif güç tüketimini doğru bir şekilde tahmin eden bir makine öğrenimi modeli geliştirmek ve bunu etkileşimli bir web uygulaması (Streamlit) aracılığıyla sunmaktır.
* **Veri Seti:** Projemizde, enerji tüketimi (aktif güç) ve çeşitli hava durumu verilerini (sıcaklık, basınç, nem, rüzgar hızı/yönü, hava durumu açıklaması) içeren bir veri seti kullanılmıştır.
""")

# Santral görselini ekleme: Eğer santral.png bulunamazsa placeholder kullan
if os.path.exists('santral.png'):
    st.image("santral.png", caption="Enerji Santrali Örneği", use_column_width=True)
else:
    st.image("https://placehold.co/800x450/333333/FFFFFF?text=Enerji%20Santrali%20Görseli", caption="Enerji Santrali Örneği (Görsel Bulunamadı)", use_column_width=True)

st.markdown("---")

# Slayt 2: Veri Analizi ve Ön İşleme
st.header("2. Veri Analizi ve Ön İşleme")
st.markdown("""
* **Veri Seti Keşfi:** Veri setinin sütun yapıları, veri tipleri detaylıca incelenmiş ve eksik değer olmadığı tespit edilmiştir. `date` sütunu model için gereksiz olduğu için çıkarılmıştır.
* **Aykırı Değer Analizi:** Sayısal sütunlarda IQR (Çeyrekler Arası Aralık) metodu kullanılarak potansiyel aykırı değer sınırları belirlenmiştir.
* **Ön İşleme Adımları:**
    * **Kategorik Veri Dönüşümü:** `description` (hava durumu açıklaması) sütunu, modelin anlayabileceği sayısal formata dönüştürülmek üzere One-Hot Encoding yöntemiyle işlenmiştir.
    * **Özellik Ölçeklendirme:** Model performansını artırmak ve algoritmaların doğru çalışmasını sağlamak için **yalnızca sayısal özellikler** (`current`, `voltage`, `temp`, `pressure`, `humidity`, `speed`, `deg`) `StandardScaler` ile ölçeklendirilmiştir. One-Hot Encoded sütunlar ölçeklenmemiştir.
""")
st.subheader('Sıcaklık Aralıkları Arasında Nem Dağılımı')
try:
    st.image('sicaklik_nem_dagilimi.png', caption='Veri Setindeki Sıcaklık ve Nem İlişkisi (Kutu Grafiği)', use_column_width=True)
except FileNotFoundError:
    st.warning("Görsel 'sicaklik_nem_dagilimi.png' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")

st.subheader('Sıcaklık ve Nem İlişkisi')
try:
    st.image('sicaklik_nem_dagilimi_scatter.png', caption='Veri Setindeki Sıcaklık ve Nem İlişkisi (Dağılım Grafiği)', use_column_width=True)
except FileNotFoundError:
    st.warning("Görsel 'sicaklik_nem_dagilimi_scatter.png' bulunamadı. Lütfen model eğitim dosyasını çalıştırdığınızdan emin olun.")
st.markdown("---")

# Slayt 3: Model Geliştirme Yaklaşımı
st.header("3. Model Geliştirme Yaklaşımı")
st.markdown("""
* **Kullanılan Regresyon Modelleri:**
    * **Temel Modeller:** Linear Regression, Ridge, Lasso (Doğrusal Modeller).
    * **Ağaç Tabanlı Modeller:** Random Forest Regressor, LightGBM, XGBoost (Daha karmaşık ve güçlü modeller).
    * **Hiperparametre Optimizasyonu:** `GridSearchCV` kullanılarak her model için en iyi performans parametreleri (örneğin ağaç sayısı, öğrenme hızı) sistematik bir şekilde belirlenmiştir.
* **Ensemble (Topluluk) Öğrenmesi:**
    * **Voting Regressor:** Birden fazla modelin tahminlerini birleştirerek daha kararlı ve doğru sonuçlar elde etmek için denenmiştir.
    * **Stacking Regressor:** Temel modellerin (Random Forest, LightGBM, XGBoost) tahminlerinin, `Ridge` regresör gibi bir meta-model tarafından yeni özellikler olarak kullanıldığı gelişmiş bir topluluk yöntemidir. **Uygulamada, en iyi performansı gösteren bu model kullanılmıştır.**
""")
st.markdown("---")

# Slayt 4: Model Performansı ve Sonuçlar
st.header("4. Model Performansı ve Sonuçlar")
st.markdown("""
* **Model Performans Metrikleri:**
    * **MAE (Mean Absolute Error - Ortalama Mutlak Hata):** Tahminlerin gerçek değerlerden ortalama sapması.
    * **MSE (Mean Squared Error - Ortalama Karesel Hata):** Büyük hataları daha çok cezalandıran hata ölçüsü.
    * **R² Skoru:** Modelin bağımlı değişkendeki varyansın ne kadarını açıkladığını gösterir ($0$ ile $1$ arasında, $1$ en iyi).
* **Modellerin Karşılaştırılması:** Tüm modellerin MAE, MSE ve R² skorları detaylıca karşılaştırılmış, Ensemble (Stacking) modelimiz en yüksek performansı göstermiştir.
* **Overfitting (Aşırı Uyum) Kontrolü:**
    * Eğitim ($0.98$) ve test ($0.98$) $R^2$ skorları arasındaki yakınlık, modelin genellenebilirliğine işaret etmektedir.
    * **K-Katlı Çapraz Doğrulama (Cross-Validation):** Modeli 5 farklı veri alt kümesi üzerinde test ederek, performansın tüm katlarda tutarlı ve yüksek kalması ($R^2$ ortalaması $\approx 0.98$, standart sapma $\approx 0.01$), modelin yeni verilere de iyi adapte olabildiğini ve aşırı uyum göstermediğini doğrulamıştır.
""")
st.markdown("---")

# Slayt 5: Enerji Tahmin Uygulaması (Streamlit Demosu)
st.header("5. Enerji Tahmin Uygulaması (Etkileşimli Demo)")
st.write("""
    Geliştirdiğimiz bu interaktif web uygulaması, modelimizin pratik kullanımını ve tahmin yeteneğini göstermektedir.
    Aşağıdaki bölümde, istediğiniz parametreleri girerek aktif güç tüketimi için anında tahminler alabilirsiniz.
""")
st.markdown("---")

# --- Uygulama Kısmı ---
st.subheader('Aktif Güç Tahmini Yapın')

col1, col2, col3 = st.columns(3)

with col1:
    current = st.number_input('Akım (Current)', min_value=0.0, value=2.53, format="%.2f") # İlk satır değerleri varsayılan yapıldı
    voltage = st.number_input('Voltaj (Voltage)', min_value=0.0, value=122.20, format="%.2f")
    temp = st.number_input('Sıcaklık (°C)', min_value=-50.0, value=24.19, format="%.2f")

with col2:
    pressure = st.number_input('Basınç (hPa)', min_value=0.0, value=1013.00, format="%.2f")
    humidity = st.number_input('Nem (%)', min_value=0.0, max_value=100.0, value=39.00, format="%.2f")
    speed = st.number_input('Rüzgar Hızı (m/s)', min_value=0.0, value=0.00, format="%.2f")

with col3:
    deg = st.number_input('Rüzgar Yönü (°)', min_value=0.0, max_value=360.0, value=0.00, format="%.2f")
    
    # Tüm benzersiz description değerlerini al (kaydedilenden yüklendi)
    # Varsayılan değeri 'clear sky' olarak ayarla
    default_description_index = all_descriptions.index('clear sky') if 'clear sky' in all_descriptions else 0
    description = st.selectbox('Hava Durumu Açıklaması (Description)', all_descriptions, index=default_description_index)


# Tahmin Yap butonu
if st.button('Aktif Güç Tahmin Et'):
    # Kullanıcı girdilerini bir DataFrame'e dönüştür
    input_data = pd.DataFrame([[current, voltage, temp, pressure, humidity, speed, deg, description]],
                                  columns=['current', 'voltage', 'temp', 'pressure', 'humidity', 'speed', 'deg', 'description'])

    # Girdiye one-hot encoding uygula
    input_encoded = pd.get_dummies(input_data, columns=['description'], dtype='int')

    # Eğitimde kullanılan tüm sütunları içerdiğinden emin ol (eksik sütunları 0 ile doldur)
    final_input = input_encoded.reindex(columns=original_X_columns, fill_value=0)

    # Debug çıktısı: Tahmin için hazırlanan nihai DataFrame'i göster
    st.write("Final Input DataFrame (Özellik İsimleri ile - Öncesi Ölçekleme):")
    st.dataframe(final_input)

    # Sadece sayısal özellikleri ölçekle
    final_input[numerical_features] = scaler.transform(final_input[numerical_features])

    # Debug çıktısı: Modele girmeden önceki ölçeklenmiş DataFrame'i göster
    st.write("Ölçeklenmiş Input DataFrame (Tahmin İçin - Sayısal Özellikler Ölçeklendi):")
    st.dataframe(final_input)

    # Tahmin yap (kaydedilen stacking regressor'ı kullanarak)
    prediction = lr_model.predict(final_input)[0]

    st.subheader('Tahmin Edilen Aktif Güç:')
    st.success(f'{prediction:.2f} kW')

st.markdown("---")

# Slayt 6: Sonuç ve Gelecek Adımlar
st.header("6. Sonuç ve Gelecek Adımlar")
st.markdown("""
* **Projenin Temel Çıkarımları:**
    * Hava durumu ve elektrik parametrelerinin aktif güç tüketimi tahmini için yüksek derecede açıklayıcı olduğunu gösterdik.
    * Ensemble (Stacking) öğrenme yaklaşımı ile çok yüksek doğrulukta tahminler elde edildi.
    * Geliştirilen Streamlit uygulaması, modelin pratik kullanımını kolaylaştırmaktadır.
* **Elde Edilen Başarılar:**
    * $0.98$ gibi yüksek ve genellenebilir bir $R^2$ skoru.
    * Tamamen işlevsel ve kullanıcı dostu bir tahmin uygulaması.
* **Gelecekteki Potansiyel Geliştirmeler:**
    * Daha fazla ve çeşitli veri entegrasyonu (örneğin, zaman serisi özellikleri, takvim bilgileri).
    * Diğer gelişmiş makine öğrenimi modellerinin veya derin öğrenme mimarilerinin denenmesi.
    * Gerçek zamanlı veri akışı entegrasyonu.
    * Modelin performans izlemesi ve otomatik yeniden eğitim mekanizmaları.
""")
st.markdown("---")
st.subheader("Teşekkürler!")
st.write("Sorularınız varsa alabilirim.")
