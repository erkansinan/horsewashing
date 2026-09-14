# Türkiye At Yarışı Tahmin Yazılımı

## Güncel Proje Sözleşmesi

Bu proje, TJK'nin günlük yarış programı, geçmiş koşu kayıtları ve idman
verilerinden yararlanarak yarış içi göreli olasılıklar üreten bir analiz
uygulamasıdır. Çıktılar kesin sonuç veya kazanç garantisi değildir; gerçek para
bahsi otomatik olarak oynanmaz.

Güncel teknoloji ve çalışma ortamı:

- Python 3.9+; proje yerel `.venv` ile çalıştırılır.
- FastAPI + Jinja2 web arayüzü ve Typer CLI.
- `httpx`, BeautifulSoup/lxml, pandas, NumPy, scikit-learn ve joblib.
- SQLite TTL cache, rate limiting ve yapılandırılmış logging.
- Veri kaynağı `RaceDataSource` sözleşmesi üzerinden adapter olarak izole edilir.

## 1. TJK Veri Kaynağı

### Kaynaklar

- [x] TJK resmi HTML sayfaları üzerinden günlük programı çek.
- [x] Günlük sonuçları ve tarih bazlı geçmiş koşuları çek.
- [x] At koşu/geçmiş bilgilerini at bazlı sayfalardan al.
- [x] Kariyer, son yıl, jokey-at kombinasyonu ve idman verilerini topla.
- [ ] TJK'nin resmi olmayan REST API'si bulunmadığı varsayımını düzenli aralıklarla yeniden doğrula.
- [ ] `github.com/SezerFidanci/TJK-API` endpoint desenlerini yeniden kontrol et.
- [ ] `github.com/fatihbozdag/Ganyan` projesini yalnızca mimari referans olarak incele; kod veya veri kopyalama.

### Veri erişim kuralları

- [x] `RaceDataSource` Protocol/adapter katmanı kullan.
- [x] User-Agent gönder.
- [x] İstekleri rate-limit et; gereksiz paralel isteklerden kaçın.
- [x] SQLite TTL cache kullan.
- [x] Canlı TJK erişimi başarısız olduğunda açık hata veya kontrollü fallback göster.
- [x] Eksik günlük programı sessizce başarılı sayma; tarih ve hipodrom bilgisini hata mesajına dahil et.

### Günlük program çıktısı

- [x] Seçilen tarih ve hipodromlara göre yarışları listele.
- [x] Hipodrom, koşu numarası, saat, mesafe, pist, grup/ikramiye ve atları göster.
- [x] Şehir/hipodrom filtresi ve yarış seçimi sağla.
- [x] At adı, jokey, antrenör, kilo, kulvar, sınıf ve ekipman gibi metin metadata'sını koru.
- [x] Metin alanlarını keyfi sayısal kodlara dönüştürme.

## 2. Zaman Güvenli TJK Özellikleri

Güncel stage-1 feature sözleşmesi `src/atyaris/ml/features.py` içindeki
`TJK_STAGE1_FEATURE_COLUMNS` sabitidir. `TJK_FEATURE_COLUMNS`, geriye dönük
uyumluluk için aynı listenin takma adıdır.

### Stage-1: temel model feature'ları

- [x] Program feature'ları: `draw`, `weight`, `distance`, `field_size`, `age`, `handicap_points`.
- [x] Kariyer ve son 365 gün toplamları.
- [x] Jokey-at kombinasyonu toplamları.
- [x] Son 3/5/10 koşu formu, varyans, son performans ve trend.
- [x] Son koşudan geçen süre, dinlenme/yorgunluk ve yarış sıklığı.
- [x] Mesafe, pist ve hipodrom uyumu.
- [x] Geçmiş koşu bitiş, alan büyüklüğü, kilo, ganyan, HP, derece, ikramiye ve S20 özetleri.
- [x] İdman sayısı, ortalama/en iyi derece, mesafe ve son idmandan geçen süre.
- [x] `history_missing`, `career_summary_missing`, `workout_missing`, `age_missing`, `handicap_missing`, `odds_missing` sinyalleri.
- [x] Kullanılmayan veya eksik geçmiş için default değerleri açık missing flag ile ayır.

### Stage-1'den çıkarılan piyasa alanları

- [x] `odds` stage-1 feature listesinde bulunmaz.
- [x] `market_probability_norm` stage-1 feature listesinde bulunmaz.
- [x] `implied_probability` stage-1 feature listesinde bulunmaz.
- [x] Bu alanları yalnızca stage-2 piyasa modeli ve EV/Kelly katmanı kullanır.

### Zaman sızıntısı kuralları

- [x] Hedef yarışın `finish_position`/`is_winner` sonucu feature üretiminde kullanılmaz.
- [x] Geçmiş koşu satırları yalnızca `race_date < target_race_date` koşuluyla kullanılır.
- [x] Kariyer ve son 365 gün özetlerini TJK'nin güncel aggregate değerlerinden doğrudan alma; filtrelenmiş geçmişten yeniden hesapla.
- [x] İdman kayıtlarını `workout_date < target_race_date` koşuluyla filtrele.
- [x] Gelecek tarihli veya hedef yarış sonrası kayıtları walk-forward eğitiminden çıkar.

## 3. Benter ML Mimarisi

### Stage-1 temel model

- [x] Yarış içi softmax kullanan conditional logit modeli uygula.
- [x] L1/L2 regularization ve standardizasyon desteği sağla.
- [x] Bağımsız ikili sınıflandırma yerine aynı yarıştaki atların göreli olasılıklarını üret.
- [x] Stage-1 eğitiminde piyasa/ganyan feature'larını kullanma.

### Stage-2 piyasa birleştirme

- [x] Stage-1 olasılığını logit sinyaline dönüştür.
- [x] Yarış içinde normalize edilmiş piyasa olasılığını logit sinyaline dönüştür.
- [x] Stage-1 logit, piyasa logit ve etkileşim terimini ikinci conditional logit modeline ver.
- [x] Tahmin sırasında gerçekten stage-2 modelini çağır; sabit `%75/%25` blend kullanma.
- [x] Eski artifact ağırlık alanlarını yalnızca serialization uyumluluğu için koru; karar hesabında kullanma.

### Loss, calibration ve olasılık çıktısı

- [x] Eğitim/validation loss olarak yarış bazlı conditional negative log-likelihood kullan.
- [x] Kazananı olmayan bozuk yarış gruplarını güvenli biçimde ele al.
- [x] Kalibratörü stage-1 yerine final stage-2 ham olasılıkları üzerinde fit et.
- [x] Tahminde final stage-2 olasılıklarını kalibre et.
- [x] Kalibre edilmiş olasılıkları yarış içinde yeniden normalize et.
- [x] Harville ile 2., 3. ve Top-3 olasılıklarını üret.

### EV, Kelly ve raporlama

- [x] `EV = calibrated_probability * odds - 1` hesabını kullan.
- [x] Fractional Kelly ve üst bahis oranı sınırı uygula.
- [x] Minimum edge, minimum EV ve minimum olasılık filtrelerini config'ten al.
- [x] Model ve piyasa karşılaştırmasını raporla.
- [x] Yarış içi olasılık toplamlarının 1 olmasını doğrula.

## 4. Web ve CLI

- [x] Ana web bağlantısının metnini `Ana sayfa` olarak göster; kök linki koru.
- [x] Günlük yarış ve at bilgilerini web arayüzünde göster.
- [x] ML tahmin üretme butonunu model, tarih, hipodrom ve prediction feature akışına bağla.
- [x] Eğitim işlemini arka planda çalıştır; ilerleme, duraklatma, iptal ve retry durumlarını göster.
- [x] Eğitim tamamlandığında artifact'in beklenen stage-1 feature listesiyle eşleştiğini kontrol et.
- [x] Eski artifact'i sessizce kullanma; eksik/eski feature listesini açıkça raporla ve yeniden eğitim iste.
- [x] Web tahmin hatalarını kullanıcıya anlaşılır Türkçe mesajla göster.
- [x] Çıktıda sıralama, güven, `P(win)`, `P(2.)`, `P(3.)`, `P(Top3)`, edge, EV, Kelly ve karar alanlarını göster.
- [x] Sonuçların kesinlik taşımadığını ve sorumlu bahis uyarısını göster.

## 5. Doğrulama ve Test

- [x] Scraper HTML parser testleri mock fixture'larla çalışır.
- [x] Cache, HTTP client, modeller, scoring ve web testleri bulunur.
- [x] Feature testleri yeni TJK stage-1 listesini ve missing flag'leri doğrular.
- [x] Stage-1 feature listesinde piyasa kolonlarının bulunmadığını test et.
- [x] Stage-2 tahmininin sabit blend yerine stage-2 model çıktısını kullandığını test et.
- [x] Conditional NLL'yi elle hesaplanan yarış örneğiyle test et.
- [x] Hedef tarih filtrelemesi ve gelecek idman/kariyer verisi sızıntısını test et.
- [x] Final stage-2 calibration sırasını test et.
- [x] Artifact kaydetme/yükleme ve eski artifact hata mesajını test et.
- [x] Walk-forward backtest çalıştır.
- [x] Log-loss, Brier, ECE/calibration, Top-1, ROI, correct-bet ratio ve model-piyasa farkını ölç.
- [x] Model-piyasa ROI farkı için bootstrap güven aralığı raporla.
- [ ] Canlı TJK erişimine bağlı olmayan tam regresyon testinin tüm 96 testte tamamlandığını ayrıca kaydet.
- [ ] Büyük gerçek TJK veri setiyle uzun dönem performans ve kalibrasyon raporu üret.

## 6. Teslim ve İşletim

- [x] `README.md` kurulum, kullanım, TJK veri envanteri ve feature anlamlarını açıklar.
- [x] `requirements.txt` ve `pyproject.toml` günceldir.
- [x] CLI ile ingest, eğitim, tahmin, backtest, rapor ve ticket optimization komutları sağlanır.
- [x] Model artifact'i `models/phase1_logreg.joblib` altında kaydedilir.
- [x] Model sağlık raporu `.health.json` olarak üretilir.
- [x] Eğitim ve tahmin yollarında atomik dosya yazımı kullanılır.
- [x] TJK erişim hataları loglanır ve kontrollü fallback uygulanır.
- [ ] TJK endpoint değişiklikleri için adapter parser sözleşmesi ve fixture güncelleme prosedürü ekle.
- [ ] Üretim çalıştırma/runbook dokümanına model yeniden eğitim ve eski artifact yenileme adımlarını ekle.

## 7. Güvenlik ve Kapsam Sınırları

- [x] TJK'ya makul istek sıklığıyla eriş.
- [x] Cache ve User-Agent kullan.
- [x] Kullanım şartlarına uygun scraping uygula.
- [x] Otomatik bahis oynama, hesap erişimi veya ödeme işlemi kapsam dışıdır.
- [x] Uygulama tahmin aracıdır; kesin sonuç ve kazanç garantisi vermez.

## İlgili Modüller

- Veri: `src/atyaris/data_sources/`, `src/atyaris/cache/`
- Feature üretimi: `src/atyaris/ml/features.py`, `src/atyaris/ml/real_ingestion.py`
- Stage-1: `src/atyaris/ml/fundamental_model.py`
- Stage-2: `src/atyaris/ml/market_blend.py`
- Calibration: `src/atyaris/ml/calibration.py`
- Harville: `src/atyaris/ml/harville.py`
- EV/Kelly: `src/atyaris/ml/ev_kelly.py`
- Pipeline: `src/atyaris/ml/pipeline.py`, `src/atyaris/ml/backtest.py`
- Web: `src/atyaris/web/app.py`, `src/atyaris/web/templates/`
- Testler: `tests/`
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