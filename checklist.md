# GÖREV: Türkiye At Yarışı Tahmin Yazılımı Geliştirme

## ROL
Sen kıdemli bir **Python yazılım mühendisisin** ve aynı zamanda **veri odaklı at yarışı analistisin**. Görevin, aşağıda tarif edilen özellikleri taşıyan, üretime hazır kalitede bir Python uygulaması geliştirmek. Kod yazarken temiz mimari, tip güvenliği (type hints), hata yönetimi ve test edilebilirlik ilkelerine uy.

## PROJENİN AMACI
Türkiye'de o gün koşulacak at yarışlarını canlı olarak çekip, kullanıcının seçtiği yarışlar için atların geçmiş istatistiklerine dayanan, gerekçelendirilmiş bahis tahminleri üreten bir sistem.

---

## 1. VERİ KAYNAKLARI (API ARAŞTIRMASI)

Türkiye'de at yarışçılığının tek resmi otoritesi **Türkiye Jokey Kulübü (TJK)**'dür (tjk.org). TJK'nın halka açık, resmi/belgelenmiş bir REST API'si **yoktur**; ancak aşağıdaki kaynaklar kullanılabilir. Ajan, uygulamayı yazmadan önce bunları doğrulayıp güncel durumlarını teyit etmeli:

1. **TJK-API (community/açık kaynak PHP sarmalayıcı)** — `github.com/SezerFidanci/TJK-API`
   TJK'nın resmi sitesinden anlık olarak yarış bülteni ve yarış sonuçlarını çeken bir sarmalayıcı. PHP ile yazılmış; mantığı incele (hangi TJK endpoint'lerini, hangi parametrelerle çağırdığını) ve aynı endpoint'leri **Python'da `requests`/`httpx` ile yeniden uygula**. Sunduğu fonksiyonlar: bugünün yarışları, tarihe göre yarışlar, bugünün sonuçları, tarihe göre sonuçlar.

2. **TJK resmi web sayfaları (HTML/JSON çıktısı — scraping/parsing gerekebilir)**
   - Günlük Yarış Programı: `tjk.org/TR/Yarissever/Info/Page/GunlukYarisProgrami`
   - Günlük Yarış Sonuçları: `tjk.org/TR/YarisSever/Info/Page/GunlukYarisSonuclari`
   - Yıllık Yarış Programı: `tjk.org/TR/YarisSever/Query/Page/YillikYarisProgrami`
   - At Koşu Bilgileri (at bazlı geçmiş performans/istatistik): `tjk.org/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId={AT_ID}` — bu sayfa, bir atın geçmiş koşu geçmişini (antrenör, jokey, kilo, derece, sıra, pist, mesafe, şehir, tarih, ganyan) tablo halinde döner. At bazlı istatistik kaynağı olarak bu kritik.
   - Jokey İstatistikleri: `tjk.org/TR/map/Query/Page/JokeyIstatistikleri`

   Bu sayfalar çoğunlukla sunucu taraflı render edilen HTML tabloları döndürüyor; resmi bir JSON API değiller. Ajan bu nedenle **BeautifulSoup / lxml ile HTML parsing** katmanı kurmalı ve TJK'nın kullanım şartlarına uygun, makul istekte bulunma sıklığı (rate limiting, caching) uygulamalı.

3. **Referans açık kaynak proje**: `github.com/fatihbozdag/Ganyan` — TJK verilerini analiz ederek makine öğrenmesi ve Bayesian yöntemlerle at yarışı tahmini yapan bir Python projesi. Mimari fikir almak için incelenebilir (scraper/analysis/utils klasör yapısı).

**Ajana talimat:** Kodlamaya başlamadan önce bu üç kaynağı (özellikle TJK-API reposunun kaynak kodunu) inceleyip hangi endpoint/URL desenlerinin hâlâ çalıştığını doğrula; TJK sitesi zaman zaman URL yapısını değiştirebiliyor. Resmi bir API bulunamazsa, **HTML scraping + response caching (örn. 5-10 dk TTL)** yaklaşımını kullan ve bunu kod içinde `data_source` soyutlama katmanıyla (interface/adapter pattern) izole et — böylece ileride resmi bir API çıkarsa sadece adapter değişir.

---

## 2. FONKSİYONEL GEREKSİNİMLER

### 2.1 Günlük Yarış Bülteni Çekme
- Uygulama açıldığında, o günkü (veya kullanıcının seçtiği tarihteki) Türkiye'deki tüm hipodromlarda (İstanbul/Veliefendi, Ankara, İzmir/Şirinyer, Bursa/Osmangazi, Adana, Antalya, Elazığ, Diyarbakır, Kocaeli, Şanlıurfa vb.) koşulacak yarışları listelemeli.
- Kullanıcının konumuna (veya seçtiği şehre) göre **"en yakın" hipodromu** önceliklendirmeli — konum bilgisi kullanıcıdan manuel seçim olarak alınabilir (GPS zorunlu değil).
- Her yarış için: hipodrom, koşu no, saat, mesafe, pist tipi (kum/çim), grup/ikramiye bilgisi, katılan atların listesi gösterilmeli.

### 2.2 Yarış Seçimi
- Kullanıcı listeden bir veya birden fazla yarış seçebilmeli (CLI menü, basit web arayüzü veya API endpoint — teknoloji tercihini aşağıdaki "Teknik Yığın" bölümüne göre belirle).

### 2.3 At İstatistiklerinin Çekilmesi
Seçilen yarıştaki her at için aşağıdaki verileri topla:
- Son 5-10 koşusunun dereceleri/sıralaması
- Galibiyet/plase/tabela oranı (son 1 yıl ve kariyer geneli)
- Mevcut jokey ve jokeyin bu at ile geçmiş performansı
- Antrenör istatistikleri
- Mesafe/pist tipi bazlı performans (bu at bu mesafede/pistte daha önce nasıl koştu)
- Kilo (bu koşudaki taşıyacağı kilo, geçmişteki kilo performansına kıyasla)
- Son koşudan bu yana geçen süre (dinlenme/form durumu göstergesi)
- Ganyan oranı (varsa açık oranlar — piyasanın atı nasıl fiyatladığının bir göstergesi)

### 2.4 Tahmin Motoru
- Toplanan istatistikleri bir **at yarışı uzmanı gibi yorumlayarak** tahmin üret: kazanan aday, plase adayları, sürpriz olabilecek at, kaçınılması önerilen favoriler.
- Tahmin gerekçesini **doğal dilde, madde madde açıkla** (örn. "Bu at son 3 koşuda bu mesafede birinci oldu, aynı jokeyle 4. kez koşuyor ve kilosu geçen seferkinden 1.5 kg hafif — form yükselişte").
- Sayısal bir **güven skoru** (0-100) veya olasılık dağılımı üret; bunu basit bir ağırlıklı puanlama modeliyle yap (örn. son form %30, jokey/antrenör kombinasyonu %20, mesafe/pist uyumu %20, kilo %15, dinlenme süresi %15 gibi — ağırlıkları config'te değiştirilebilir yap).
- İsteğe bağlı: ileri seviye bir sürüm için scikit-learn ile geçmiş sonuçlar üzerinden eğitilmiş basit bir sınıflandırma/regresyon modeli (gradient boosting vb.) eklenebilir; bu MVP sonrası faz olarak planlanmalı.

### 2.5 Çıktı
- Her seçilen yarış için: sıralanmış at listesi + tahmini bitiş sırası + güven skoru + kısa gerekçe metni.
- Bahis türü önerisi (ganyan, plase, ikili, tabela vb.) kullanıcının isteğine göre opsiyonel olarak eklenebilir.

---

## 3. TEKNİK YIĞIN ÖNERİSİ
- **Dil:** Python 3.11+
- **HTTP/scraping:** `httpx` veya `requests` + `BeautifulSoup4`/`lxml`
- **Veri işleme:** `pandas`
- **Önbellekleme:** `diskcache` veya basit SQLite tabanlı cache (TJK'ya aşırı istek göndermemek için)
- **Config/ağırlıklar:** `pydantic` ile ayarlanabilir model + `.env`/`config.yaml`
- **Arayüz:** Başlangıç için CLI (`typer` veya `argparse`); istenirse `FastAPI` ile basit bir REST API veya `streamlit` ile hızlı bir web arayüzü
- **Test:** `pytest`, HTML/veri parsing için mock fixture'lar kullan (canlı siteye test sırasında bağımlı olma)
- **Loglama:** `logging` modülü, hata ayıklama için yapılandırılmış log

---

## 4. MİMARİ (KATMANLI YAPI)
```
/data_sources      -> TJK'dan veri çeken adapter'lar (bülten, sonuçlar, at istatistikleri)
/models             -> Pydantic veri modelleri (Race, Horse, Jockey, Trainer, RaceStats)
/prediction         -> Puanlama/ağırlıklandırma motoru, gerekçe metni üretimi
/cache              -> İstek önbellekleme katmanı
/cli veya /api       -> Kullanıcı arayüzü katmanı
/tests               -> Birim ve entegrasyon testleri
```
Ajan, `data_sources` katmanını bir **interface (Protocol/ABC)** üzerinden soyutlamalı; böylece TJK-API scraping yöntemi değişirse veya resmi bir API bulunursa yalnızca adapter implementasyonu değişsin, geri kalan kod etkilenmesin.

---

## 5. KISITLAR VE UYARILAR (AJANA ÖZEL TALİMAT)
- Bu yazılım **tahmin/analiz aracıdır**, kesin sonuç garantisi vermez. Üretilen her çıktının sonunda otomatik olarak "Bu tahminler istatistiksel analize dayanır, kesinlik taşımaz; sorumlu bahis oynayın" şeklinde bir uyarı metni eklenmeli.
- TJK verilerini çekerken **kullanım şartlarına** ve makul istek sıklığına (rate limiting, `User-Agent` belirtme, aşırı paralel istek göndermeme) dikkat edilmeli.
- Gerçek para bahsi otomasyonu (otomatik kupon oynama) bu kapsamda **değildir** — yazılım yalnızca tahmin/analiz üretir, kullanıcı bahsi kendisi TJK'nın resmi bayilerinden (atyarışı.com, TJK bayileri vb.) oynar.

---

## 6. TESLİM EDİLECEKLER
1. Çalışan Python kod tabanı (yukarıdaki mimariye uygun, tip anotasyonlu, docstring'li)
2. `README.md`: kurulum, kullanım, veri kaynağı adaptörünün nasıl güncelleneceği
3. `requirements.txt` / `pyproject.toml`
4. Örnek çalıştırma: bir gün için bülten çekme → yarış seçme → tahmin üretme uçtan uca akışının konsol çıktısı örneği
5. Temel test paketi (`pytest`)

Şimdi bu gereksinimlere göre projeyi adım adım (önce veri katmanı, sonra modeller, sonra tahmin motoru, en son arayüz) geliştirmeye başla. Her adımdan sonra kısa bir özet ver ve bir sonraki adıma geç.

## 7. ML yöntemi
   Aşağıdaki akademik olarak kanıtlanmış yöntemleri temel alan bir 6'lı ganyan 
tahmin mimarisi kur. Rastgele/keyfi bir ML yaklaşımı değil, at yarışı 
literatüründeki en başarılı ve gerçek parayla test edilmiş yöntemleri uygula:

1. ÇEKİRDEK MODEL — Benter Mimarisi (1994, Hong Kong'da 5 yıl kanıtlanmış kâr)
   - İki aşamalı yaklaşım kur:
     a) Aşama 1 ("Temel Model"): Koşullu lojistik regresyon (conditional 
        logistic regression / conditional logit) ile her atın "gücünü" 
        (strength) tahmin et. Girdi: form, dinlenme süresi, tempo, jokey/
        antrenör istatistikleri, mesafe/zemin geçmişi.
     b) Aşama 2 ("Piyasa Birleştirme"): Aşama 1'in çıktısını piyasa 
        oranlarının (ganyan bahis oranı) ima ettiği olasılıkla ikinci bir 
        koşullu logit modelinde birleştir. Bu, modelin piyasa bilgisinden 
        de faydalanmasını sağlar (Benter'in orijinal iyileştirmesi).
   - Veri setinde yeterli hacim varsa (Benter/Silverman-Suchard önerisi), 
     koşullu logit yerine regularize edilmiş (L1/L2 penaltili) versiyonunu 
     veya modern bir ensemble (LightGBM/XGBoost) ile hibrit kullan — ama 
     çıktı mutlaka koşullu logit'in ürettiği gibi YARIŞ İÇİ göreli 
     olasılıklar (bir yarıştaki tüm atların olasılıkları toplamı = 1) 
     olmalı, bağımsız ikili sınıflandırma değil.

2. ÇOKLU SIRALAMA OLASILIKLARI — Harville Formülü
   - Kazanma olasılıklarından, 2. ve 3. sıra bitirme olasılıklarını türetmek 
     için Harville formülünü uygula (plaselı/ikili/üçlü bahisler için gerekli).
   - Gerekirse ardışık çıkarma (sequential elimination) mantığıyla genişlet.

3. KALİBRASYON
   - Ham model çıktısını gerçek frekanslarla eşleştirmek için Platt scaling 
     veya isotonic regression uygula. Benter ve sonraki çalışmaların 
     vurguladığı gibi, kalibrasyonsuz olasılıklar EV hesaplamasını bozar.

4. EV VE BAHİS BOYUTU
   - EV = (kalibre_olasılık × piyasa_oranı) - 1
   - Fractional Kelly Criterion ile bahis büyüklüğü öner (tam Kelly yerine 
     — literatürde varyans riskini azaltmak için standart pratik)
   - Sadece piyasa oranının modelin ima ettiği olasılıktan anlamlı şekilde 
     saptığı (value bet) durumları filtrele

5. ÖZELLİK ÖNCELİKLENDİRME (Borowski ve ark. 2021 bulgularına göre)
   - En yüksek prediktif ağırlığı geçmiş performans/kazanç metriklerine ver
   - Jokey/antrenör özelliklerini ikincil önem sırasına koy (literatürde 
     etkileri daha zayıf çıktı — ama tempo uyumu ve dinlenme süresi gibi 
     BAĞLAMSAL etkileşim terimleri olarak dahil et)

6. DOĞRULAMA
   - Walk-forward (zaman bazlı) backtest kur
   - Metrikler: log-loss, calibration curve, gerçek ROI simülasyonu, 
     doğru bahis oranı (correct bet ratio)
   - Modelin çıktısını ham piyasa oranlarıyla (kalibre edilmemiş favori 
     sıralaması) karşılaştırarak gerçek bir edge (kenar/avantaj) olup 
     olmadığını istatistiksel olarak test et (ör. bootstrap güven aralığı)

Kodu modüler yaz: fundamental_model.py (Aşama 1), market_blend.py (Aşama 2), 
harville.py, calibration.py, ev_kelly.py, backtest.py. Her modülde hangi 
akademik kaynağa dayandığını (Benter 1994, Harville formülü vb.) yorum 
olarak belirt.