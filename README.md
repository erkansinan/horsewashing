# Turkiye At Yarisi Tahmin Yazilimi (atyaris)

Turkiye'de o gun kosulacak at yarislarini listeleyen, kullanicinin sectigi
yarislar icin atlarin gecmis istatistiklerine dayanan, gerekceli bahis
tahminleri ureten bir Python uygulamasi.

> Bu yazilim bir **tahmin/analiz aracidir**, kesin sonuc garantisi vermez.
> Uretilen her cikti sonunda otomatik olarak bir uyari metni eklenir:
> *"Bu tahminler istatistiksel analize dayanir, kesinlik tasimaz; sorumlu
> bahis oynayin."*

## Mimari

```
src/atyaris/
  config.py           -> Ayarlar (skorlama agirliklari, cache/HTTP parametreleri)
  services.py          -> CLI ve web arayuzunun paylastigi ortak is mantigi (data source secimi, tarih parse, bulten cekme)
  models/             -> Pydantic veri modelleri (Race, RaceEntry, HorseStatistics, ...)
  data_sources/        -> Veri kaynagi adapter'lari (Protocol/Adapter deseni)
    base.py             -> RaceDataSource arayuzu (Protocol)
    sample_source.py    -> Deterministik ornek/demo veri kaynagi
    tjk_scraper.py       -> Canli TJK HTML scraping adapter'i (bkz. "Bilinen Kisitlar")
    http_client.py       -> Hiz sinirlamali (rate-limited) HTTP istemcisi
  cache/                -> SQLite tabanli TTL onbellek
  prediction/           -> Puanlama motoru, gerekce uretimi, orkestrasyon
    scoring.py           -> Agirlikli 0-100 skor hesaplama
    reasoning.py         -> Dogal dilde gerekce metinleri + etiket
    engine.py            -> PredictionEngine (orkestrasyon)
  cli/                  -> Typer tabanli komut satiri arayuzu (birincil arayuz)
  web/                  -> FastAPI + Jinja2 tabanli basit HTML arayuzu (opsiyonel, tarayicidan kullanim icin)
    app.py                -> FastAPI route'lari (/, /predict)
    templates/            -> Jinja2 HTML sablonlari
tests/                  -> pytest test paketi (canli siteye bagimli degil)
```

`data_sources` katmani bir **Protocol (yapisal arayuz)** uzerinden
soyutlanmistir (`RaceDataSource`). Uygulamanin geri kalani (tahmin motoru,
CLI) yalnizca bu arayuze bagimlidir; TJK scraping yontemi degisirse veya
resmi bir API cikarsa yalnizca yeni bir adapter yazilir, geri kalan kod
etkilenmez.

## Bilinen Kisitlar (onemli)

TJK'nin (tjk.org) herkese acik, resmi bir REST API'si yoktur. Bu proje iki
veri kaynagi adapter'i icerir:

1. **`SampleDataSource`** (varsayilan): Tamamen deterministik, gercekci
   Turkce isimlerle uretilmis ornek/demo veri. Ag baglantisi gerektirmez,
   testlerde ve uctan uca demo akisinda kullanilir.
2. **`TJKHtmlDataSource`** (deneysel, garantisiz): tjk.org'un
   `GunlukYarisProgrami` sayfasini BeautifulSoup ile ayristirmayi dener.
   **Onemli bulgu**: TJK bu sayfayi buyuk olcude JavaScript/jQuery
   unobtrusive-ajax ile render eder (sunucu once bos bir "kabuk" doner,
   asil kosu tablosu tarayicida calisan JS'in tetikledigi ic ice AJAX
   cagrilariyla sonradan doldurulur). Bu nedenle:
   - Basit bir HTTP GET + BeautifulSoup ile canli bultenden guvenilir
     sekilde kosu verisi cekmek her zaman mumkun olmayabilir; adapter,
     hicbir kosu bulamadiginda bunu acikca loglar (bkz.
     `tjk_scraper.py` ust kismindaki "UYARI" notu).
   - `--city`/`--near` parametreleri `KNOWN_HIPPODROMES` listesindeki
     resmi hipodrom adlariyla, Turkce karakter/buyuk-kucuk harf
     farkindan bagimsiz olarak eslestirilir (`istanbul`, `İstanbul`,
     `ISTANBUL` hepsi calisir).
   - At bazli gecmis performans sayfasi (`AtKosuBilgileri`) da benzer
     sekilde istemci tarafinda (AJAX/JS) yuklenir; bu nedenle
     `get_horse_statistics` bu adapter'da desteklenmiyor ve acik bir
     `DataSourceError` firlatir.
   - Tam guvenilir canli veri icin (headless tarayici ile JS calistirma
     veya butun ic ice AJAX zincirinin tersine muhendisligi gibi) daha
     kapsamli bir yaklasim gerekir; bu, projenin mevcut kapsaminin
     disindadir.

Bu kisitlar nedeniyle **varsayilan ve onerilen CLI/web kaynagi `sample`'dir**.
`--source tjk` / `?source=tjk` deneysel bir secenektir; canli veri
gelmezse (bos bulten + log uyarisi) `sample`'a donun.

### Veri Kaynagini Guncelleme

Yeni bir hipodrom eklemek veya TJK sablon degisikligine uyum saglamak icin:
1. `https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisProgrami?QueryParameter_Tarih=GG/AA/YYYY&SehirAdi=HipodromAdi`
   adresini, ilgili hipodromun TJK'daki resmi adiyla acip sayfanin dogru
   yuklendigini (ve verinin sunucu tarafinda mi yoksa JS ile mi
   render edildigini) dogrulayin.
2. Hipodrom adini `src/atyaris/data_sources/tjk_scraper.py` icindeki
   `KNOWN_HIPPODROMES` listesine ekleyin.
3. `_parse_daily_program` / `_parse_entry_row` icindeki regex/kolon
   varsayimlarini, guncel sayfa yapisina gore dogrulayin ve gerekirse
   duzeltin (canli sayfanin HTML kaynagini inceleyin).
4. `tests/test_tjk_scraper.py` icindeki sabit HTML fixture'ini yeni yapiya
   gore guncelleyip testleri calistirin.

Resmi bir TJK API'si ileride yayinlanirsa, yalnizca `RaceDataSource`
arayuzunu uygulayan yeni bir adapter yazip CLI'da kaynak olarak eklemeniz
yeterlidir.

## Kurulum

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Benter Tabanli ML Pipeline

Bu repo, literaturde denenmis iki asamali Benter mimarisini uygular:

- Asama 1 (fundamental_model): Yaris ici goreli kazanim olasiliklari ureten conditional logit.
- Asama 2 (market_blend): Asama 1 olasiliklarini piyasa (ganyan) bilgisiyle ikinci bir kosullu modelde birlestirme.
- Harville (harville.py): Kazanma olasiliklarindan 2.lik, 3.luk ve top3 olasiliklarini turetme.
- Kalibrasyon (calibration.py): Platt veya isotonic ile olasilik duzeltmesi.
- EV + Fractional Kelly (ev_kelly.py): Value bet filtreleme ve bahis boyutu onerisi.
- Walk-forward backtest (backtest.py): log-loss, calibration, ROI ve market baseline karsilastirmasi.

Asagidaki komutlar sentetik veriyle uctan uca Benter akisidir:

```powershell
python main.py ingest --start-date 2024-01-01 --end-date 2025-12-31
python main.py preprocess
python main.py features
python main.py train
python main.py predict --date 2025-12-31
python main.py backtest
python main.py optimize-ticket --date 2025-12-31 --budget 500
python main.py report --date 2025-12-31 --budget 500
```

Esdeger olarak mevcut CLI uzerinden de calistirabilirsiniz:

```powershell
atyaris ml ingest
atyaris ml preprocess
atyaris ml features
atyaris ml train
atyaris ml predict --date 2025-12-31
atyaris ml backtest
atyaris ml optimize-ticket --date 2025-12-31 --budget 500
atyaris ml report --date 2025-12-31 --budget 500
```

Not: `calibrate`, `optimize`, `report` komutlari aktiftir.

### TJK Sonuclariyla Karsilastirma ve Yeniden Egitim

Gecmis bir gunun hipodrom sonuclarini resmi TJK sayfasindan ozetlemek ve
mevcut ML tahminleriyle karsilastirmak icin:

```powershell
atyaris ml results --date 2026-09-06 --city İstanbul
```

`main.py` dogrudan kullaniliyorsa ayni komut `python main.py results ...`
seklindedir. Sonuclar, TJK'nin resmi start numaralariyla `draw` alanini
eslestirir; eksik/bitmemis kosular egitim etiketi olarak kullanilmaz.

Sonuc etiketleriyle gercek TJK verisini yeniden cekip modeli egitmek ve
walk-forward backtest almak icin:

```powershell
atyaris ml learn --start-date 2026-08-01 --end-date 2026-09-06
```

Bu komut `raw -> preprocess -> features -> train -> backtest` akisinin tamamini
calistirir ve yeni model versiyonunu deney takip veritabanina kaydeder.

### Gercek TJK egitiminde checkpoint ve devam etme

Web arayuzundeki `/training` sayfasi, secilen tarih araligini gun gun isler.
Her gun tamamlandiktan sonra su dosyalar atomik olarak guncellenir:

- `data/raw/tjk_real_races.csv`: tamamlanan gunlerin ham etiketli verisi
- `data/raw/tjk_real_races.progress.json`: tamamlanan gunlerin checkpoint'i
- `data/raw/tjk_real_races.jobs.json`: duraklatilabilir web egitim isinin durumu

Bu sayede ag kesintisi, uygulama kapanmasi veya kullanicinin duraklatmasi
sonrasinda egitim ayni tarih araliginda tamamlanan gunleri tekrar cekmeden
devam eder. Sunucu yeniden baslatildiginda yarim kalmis `running` isler
guvenli olarak `paused` durumuna alinir; `/training` sayfasindaki **Kaldigi
yerden devam et** dugmesiyle ayni job ve checkpoint kullanilarak surdurulur.

Bir gun veya hipodrom icin bulten/sonuc alinamazsa bu durum egitimi otomatik
olarak `failed` yapmaz. Ulasilabilen veriler korunur, basarisiz tarih ve
kosular tamamlanmis checkpoint olarak isaretlenmez ve daha sonra tekrar
denenebilir. Web sayfasi kullanilabilir veri varsa **Mevcut verilerle devam
et** veya **Atlananlari tekrar dene** seceneklerini gosterir. Hic kullanilabilir
veri yoksa yalnizca tekrar deneme onerilir; boylece bos veriyle model egitimi
baslatilmaz.

Web egitim akisinda:

1. `/training` sayfasindan baslangic ve bitis tarihlerini secin.
2. **Egitimi baslat** ile yeni bir job olusturun.
3. Uzun bir calismayi durdurmak icin **Egitimi duraklat** dugmesine basin.
4. Uygulama yeniden baslarsa veya daha sonra devam etmek isterseniz
  **Kaldigi yerden devam et** dugmesine basin.
5. Kalici olarak vazgecmek icin egitimi durdurun; durdurma ile duraklatma
  ayni anlama gelmez.

Ilerleme mesajlarinda `gunluk at` o anki gunun toplamini, `toplam biriken`
ise secilen tarih araliginda su ana kadar kaydedilen satir sayisini gosterir.
Toplam kalan sure tahmini; tamamlanan son gunlerin medyan suresini, aktif
gundeki `islenen/toplam at` oranini ve kalan gun sayisini birlikte kullanir.
Ilk gunun ilk bolumunde tahmin yaklasiktir; gun tamamlandikca daha kararlı
hale gelir.

TJK isteklerinde bugunun gorunumu icin `Era=today`, gecmis egitim tarihleri
icin `Era=past` kullanilir. Bos veya eksik sonuc sayfalari sessizce egitime
katilmaz; ilerleme mesajinda ilgili tarih ve hipodrom uyarilir. TJK verisi
deneysel oldugu icin egitim sonuclarini ve kaydedilen satir sayilarini
kontrol etmek gerekir.

Benter notlari:
- `train` komutu iki asamali conditional logit modeli egitir ve kalibratoru kaydeder.
- `predict` ciktilarinda `P(win)`, `P(2.)`, `P(3.)`, `P(Top3)`, `Edge`, `EV`, `Kelly`, `Decision` kolonlari bulunur.
- `backtest` ciktilari log-loss, brier, ece, roi, correct_bet_ratio ve model-vs-market ROI farki (bootstrap CI) icerir.
- `optimize-ticket` 6 ayakli kolonlari beam search + Monte Carlo ile butce altinda secer.
- `report` backtest + tahmin + kupon optimizasyonu + feature etkilerini tek JSON/HTML dashboardda birlestirir.

### Benter Icin Config Rehberi

Temel ayarlar [config.yaml](config.yaml) dosyasindadir. Benter mimarisinde aktif olarak kullanilan gruplar:

- Veri dosyalari: `phase1_raw_csv_path`, `phase1_clean_csv_path`, `phase1_features_csv_path`, `phase1_prediction_features_csv_path`, `phase1_model_path`
- Egitim bolme parametreleri: `phase1_min_train_days`, `phase1_holdout_days`, `phase3_calibration_days`
- Kalibrasyon: `phase3_calibration_method` (`isotonic`, `platt`, `none`)
- Value bet filtreleri: `ev_probability_threshold`, `ev_min_edge`, `ev_min_value`
- Kelly risk kontrolu: `ev_fractional_kelly`, `ev_max_kelly_fraction`
- Kupon optimizasyonu: `phase4_default_budget`, `phase4_beam_width`, `phase4_top_per_leg`, `phase4_simulation_count`
- Raporlama/izleme: `phase5_tracking_db_path`, `phase5_report_dir`, `phase5_top_feature_count`

Geriye uyumluluk notu:

- `ensemble_boosting_weight`, `ensemble_ranking_weight`, `calibration_temperature` alanlari yalnizca klasik skor/legacy akista anlamlidir.
- Benter ML boru hatti bu uc parametreyi kullanmaz.

## Kullanim

### Gunun bultenini listele (ornek veriyle)

```powershell
atyaris bulletin --source sample
```

### Belirli yarislar icin tahmin uret

```powershell
atyaris predict --source sample --races 1,2
```

### Interaktif mod (bulteni goster, konsoldan secim al)

```powershell
atyaris interactive --source sample
```

### Gercek TJK verisiyle (yalnizca desteklenen hipodromlar)

```powershell
atyaris bulletin --source tjk --city Ankara --date 18.08.2026
```

### "En yakin" hipodromu oncelikle goster

```powershell
atyaris bulletin --source sample --near Ankara
```

## Web Arayuzu (HTML, opsiyonel)

CLI birincil arayuzdur; ayrica tarayicidan kullanilabilen basit bir FastAPI +
Jinja2 HTML arayuzu de bulunur (checklist.md > bolum 3'teki opsiyonel "FastAPI
ile basit web arayuzu" onerisini karsilar). Ayni is mantigini (`atyaris.services`,
`PredictionEngine`) CLI ile paylasir; kod tekrari yoktur.

```powershell
atyaris web
# tarayicida ac: http://127.0.0.1:8000/
```

- Ana sayfa (`/`): kaynak (sample/tjk), tarayici takviminden secilebilen
  tarih ve o tarihte yarisi olan hipodromlarin listelendigi acilir menu
  ile gunun bultenini listeler (hipodrom, saat, mesafe, pist,
  grup/ikramiye bilgisi, katilan atlarin listesi).
- Her yaris satirindaki **"Tahmin Uret"** linki, o yaris icin siralanmis
  tahmin sayfasini (`/predict`) acar: skor tablosu, at basina gerekce
  metinleri ve zorunlu uyari metni.
- `--host`/`--port`/`--reload` secenekleriyle ozellestirilebilir:
  `atyaris web --port 8080 --reload`.

### Web'de ML (Benter) ile Tahmin

Web arayuzunde `Kaynak` alaninda `ml (benter)` secildiginde,
sayfa klasik skor motoru yerine ML pipeline ciktilarini gosterir.

ML modu icin onerilen baslatma:

```powershell
.venv\Scripts\Activate.ps1
.venv\Scripts\atyaris.exe web
```

Not: Sistemde global `atyaris` komutu varsa farkli Python ortamina gidebilir.
Bu durumda `.venv\Scripts\atyaris.exe web` kullanin.

ML modunda kullanim adimlari:

1. Ana sayfada `Kaynak = ml (benter)` secin.
2. Takvimden tarih secin.
3. (Opsiyonel) Hipodrom filtresi secin.
4. `Bulteni Getir` ile ML yaris listesini acin.
5. Ilgili satirdan `ML Tahmin Gor` ile kosu detayina gecin.

ML tahmin ekraninda sunulan basliklar:

- `P(win)`: Kalibre edilmis kazanma olasiligi.
- `P(2.)`, `P(3.)`, `P(Top3)`: Harville formulu ile turetilen siralama olasiliklari.
- `Guven`: Model-piyasa sapmasindan uretilen guven skoru.
- `Edge`, `EV`, `Kelly`: Value bet analizi ve fractional Kelly bahis buyuklugu.
- `Karar (BET/NO_BET)`: Olasilik, edge, EV ve Kelly filtrelerine gore sinyal.
- `Kupon Optimizasyon Ozet`: 6 ayakli beam search + Monte Carlo sonucu.
- Tablolarda teknik ID yerine okunur etiketler kullanilir:
  yaris satirinda `Hipodrom - N. Kosu`, at satirinda `At adi` gosterilir.

Hipodrom secimi notu:

- ML modunda sehir filtresi Turkce karakter farklarindan bagimsiz eslesir.
  Ornek: `İstanbul`, `istanbul`, `ISTANBUL` ayni secim olarak kabul edilir.

Otomatik ML hazirlik davranisi:

- Secilen tarih icin model/veri hazir degilse, web katmani gerekli
  sentetik veri + feature + train adimlarini bir kez otomatik tamamlar.
- Eski/uyumsuz model dosyasi algilanirsa model yeniden uretilir.

Sik gorulen durumlar:

- `Secili tarih icin ML kosu bulunamadi.`:
  Tarih filtrelemesi sonucunda ilgili gunde kayitli race olmayabilir.
- `ML race_id bulunamadi...`:
  Sayfa acikken tarih/hipodrom degisti ise eski linkle gidilmis olabilir;
  liste ekranindan kosuyu yeniden secin.

## Ornek Calistirma Ciktisi (uctan uca akis, gercek/dogrulanmis cikti)

Asagidaki cikti `atyaris predict --source sample --city Ankara --races 1`
komutunun gercek calistirmasindan alinmistir:

```text
$ atyaris predict --source sample --city Ankara --races 1

                      Gunun Yarislari
┏━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┓
┃ # ┃ Hipodrom ┃ Kosu ┃ Saat  ┃ Mesafe ┃ Pist ┃ At Sayisi ┃
┡━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━━┩
│ 1 │ Ankara   │ 1    │ 13:30 │ 1600m  │ Cim  │ 8         │
│ 2 │ Ankara   │ 2    │ 14:00 │ 1400m  │ Cim  │ 7         │
│ 3 │ Ankara   │ 3    │ 14:30 │ 1200m  │ Kum  │ 7         │
└───┴──────────┴──────┴───────┴────────┴──────┴───────────┘
──────────────────────── Ankara - 1. Kosu (1600m, Cim) ─────────────────────────
┏━━━━━━┳━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Sira ┃ No ┃ At           ┃ Jokey         ┃ Skor     ┃ Etiket                 ┃
┡━━━━━━╇━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ 8  │ RUZGAR GIBI  │ Serkan Yildiz │ 57.0/100 │ Kazanan Aday           │
│ 2    │ 2  │ FIRTINA KIZI │ Ahmet Celik   │ 54.4/100 │ Plase Adayi            │
│ 3    │ 4  │ KRAL YAVUZ   │ Mehmet Kaptan │ 45.4/100 │ Plase Adayi            │
│ 4    │ 7  │ DELI DOLU    │ Ahmet Celik   │ 45.1/100 │ Diger                  │
│ 5    │ 6  │ MERT YIGIT   │ Serkan Yildiz │ 40.6/100 │ Kacinilmasi Onerilen   │
│      │    │              │               │          │ Favori                 │
│ 6    │ 1  │ SUZME ELMAS  │ Ahmet Celik   │ 39.0/100 │ Diger                  │
│ 7    │ 5  │ SAHIN KANADI │ Ahmet Celik   │ 35.3/100 │ Diger                  │
│ 8    │ 3  │ GONUL ATESI  │ Onder Sahin   │ 34.9/100 │ Diger                  │
└──────┴────┴──────────────┴───────────────┴──────────┴────────────────────────┘

1. RUZGAR GIBI (Kazanan Aday, skor 57.0/100)
  - Son 7 kosunun 3 tanesinde ilk 3'e girdi (2 galibiyet dahil).
  - Serkan Yildiz, bu atla daha once 7 kez kosti ve %29 kazanma oranina sahip.
  - Bu mesafe/pist kombinasyonunda (1600m, Cim) 2 kosuda 0 galibiyet aldi.
  - Bu kosudaki kilosu gecmis ortalamasina gore 0.4 kg daha hafif — avantajli.
  - Son kosusundan bu yana 32 gun gecmis; ideal dinlenme araliginda.

2. FIRTINA KIZI (Plase Adayi, skor 54.4/100)
  - Son 8 kosunun 2 tanesinde ilk 3'e girdi (1 galibiyet dahil).
  - Ahmet Celik, bu atla daha once 8 kez kosti ve %12 kazanma oranina sahip.
  - Bu mesafe/pist kombinasyonunda (1600m, Cim) 3 kosuda 1 galibiyet aldi.
  - Bu kosudaki kilosu gecmis ortalamasina gore 0.5 kg daha agir — dezavantajli olabilir.
  - Son kosusundan bu yana 28 gun gecmis; ideal dinlenme araliginda.

... (kalan atlar icin de benzer gerekceler uretilir) ...

Bu tahminler istatistiksel analize dayanir, kesinlik tasimaz; sorumlu bahis
oynayin.
```

(Tam ciktida 8 atin tamami icin ayrintili gerekce listelenir; yukaridaki
sayilar `sample` kaynaginin deterministik ureteciyle sabittir ve her
calistirmada ayni sonucu verir.)

## Klasik Skorlama Modeli (Non-ML Yol)

Toplam skor, `config.yaml` uzerinden ayarlanabilir agirliklarla hesaplanir:

| Bilesen              | Varsayilan agirlik | Aciklama                                            |
|-----------------------|--------------------|------------------------------------------------------|
| Form                  | %30                | Son N kosunun yakinlik agirlikli bitis sirasi        |
| Jokey/Antrenor        | %20                | Jokey-at kombinasyon basarisi + kariyer kazanma orani |
| Mesafe/Pist uyumu     | %20                | Ayni mesafe (+/-200m) ve pist tipindeki performans    |
| Kilo                  | %15                | Bugunku kilo, gecmis ortalamaya kiyasla               |
| Dinlenme suresi       | %15                | Son kosudan bu yana gecen gun (ideal aralik: 14-45)   |

## Tahmin Tablosu Yorumlama Rehberi

Web tahmin tablosundaki su basliklar birlikte yorumlanmalidir:

- Kazanma Olasiligi: Modelin o atin yarisi kazanma ihtimali icin kalibre ettigi olasiliktir.
  Daha yuksek deger, goreceli kazanma sansinin daha yuksek oldugunu gosterir.
- Guven: Modelin bu olasilik tahmininden ne kadar emin oldugunu gosterir.
  Benzer olasilikta iki at varsa, Guven degeri yuksek olan tercih edilir.
- EV: Beklenen degerdir ve su sekilde hesaplanir: EV = (Kazanma Olasiligi x Ganyan) - 1.
  EV > 0 ise teorik olarak uzun vadede pozitif getiri adayi, EV < 0 ise negatif beklenti anlamina gelir.
- Kelly: Sermaye yonetimi icin onerilen bahis oranidir.
  Tablodaki deger fractional Kelly oldugundan, toplam bakiyenin tamamini degil kontrollu bir kismini onerir.

Pratik yorum sirasi:

1. Once Kazanma Olasiligi ve Guven ile modelin guclu gordugu atlari ayiklayin.
2. Sonra EV ile piyasa oranina gore gercekten "deger" olup olmadigini kontrol edin.
3. Bahis buyuklugunu Kelly degerine gore sinirlayin; Kelly dusukse ya bahis atlanir ya da cok kucuk tutulur.
4. Tek bir kolona gore karar vermeyin; bu metrikler birlikte kullanildiginda daha anlamli sonuc verir.

## Test

```powershell
pytest
```

Tum testler `SampleDataSource` veya sabit HTML fixture'lari kullanir; canli
TJK sitesine bagimli degildir.

## Bagimliliklar

Bkz. [requirements.txt](requirements.txt) ve [pyproject.toml](pyproject.toml):
`httpx`, `beautifulsoup4`, `lxml`, `pydantic`, `pydantic-settings`,
`PyYAML`, `typer`, `rich`, `fastapi`, `uvicorn`, `jinja2`, `pytest`.

## Python Surumu Notu

`pyproject.toml` `requires-python = ">=3.9"` olarak ayarlidir. Kod tabani
tamamen tip anatasyonlarinda `from __future__ import annotations` kullanir
ve Pydantic modellerinde `X | None` yerine `typing.Optional[X]` tercih
edilerek Python 3.9 ile de calisacak sekilde yazilmistir. Test paketi bu
repoda Python 3.9.1 ile calistirilip 22/22 testin gectigi dogrulanmistir.
Daha yeni bir Python (3.11+) kullaniyorsaniz herhangi bir degisiklik
gerekmez.
