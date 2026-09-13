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

### TJK'dan Cekilebilen Verilerin Tam Listesi

Asagidaki liste, mevcut `TJKHtmlDataSource` kodunun TJK sayfalarindan
ayristirip uygulama modellerine aktardigi alanlari gosterir. Bir alanin TJK
sayfasinda gorunmesi, her tarih ve hipodromda dolu olacagi anlamina gelmez;
TJK sablonu, JavaScript ile sonradan yukleme ve eksik hucreler nedeniyle alanlar
`None`, bos liste veya varsayilan deger olarak gelebilir.

#### Gunluk yaris programi (`GunlukYarisProgrami`)

`get_daily_races()` ve `get_available_hippodromes()` ile su bilgiler cekilir:

- Hipodrom adi ve o tarihte aktif hipodrom sekmeleri
- TJK `SehirId` degeri (sayfa icindeki hipodrom sekmesini bulmak icin kullanilir;
  `Race` modeline ayri alan olarak yazilmaz)
- Yaris numarasi
- Baslangic saati ve yaris tarihi
- Mesafe (metre)
- Pist turu: `Kum`, `Cim` veya `Sentetik`
- Grup/sart bilgisi: TJK yaris basligindan yakalanan ham metin
- Start numarasi
- At adi
- Atin kaynak TJK kimligi (`QueryParameter_AtId`)
- At yasi
- Jokey adi
- Antrenor adi
- Kosu kilosu
- Handikap puani (`HP`)
- Ganyan/oran
- Son kosu formu ve ham form metni
- Kosmaz durumu

Uygulamada bu bilgiler `Race` ve `RaceEntry` modellerine yazilir. TJK satirinda
bulunabilen sahip adi, takilar, cinsiyet/irk, ikramiye veya hava durumu gibi
alanlar gunluk program ayristricisinda su anda ayrica saklanmaz. `Race` modelinde
`prize_info` alani bulunmasina ragmen mevcut TJK program parser'i bu alani
doldurmamaktadir.

#### Gecmis kosu performansi (`AtKosuBilgileri`)

Atin `QueryParameter_AtId` degeri bulunuyorsa `get_horse_statistics()` ile
atin gecmis kosu satirlari ve ozet istatistikleri cekilir:

- Kosu tarihi
- Hipodrom/sehir
- Mesafe
- Pist turu
- Bitis sirasi
- Kosudaki at sayisi (`St`)
- Jokey adi
- Antrenor adi
- Taki/ekipman
- Grup bilgisi
- Kosu numarasi ve kosu adi
- Kosu cinsi/sinifi
- Sahip adi
- Handikap puani (`HP`)
- Ikramiye bilgisi
- `S20` degeri
- Tasidigi kilo
- Ganyan/oran
- Derece, saniyeye cevrilmis olarak

Gecmis kosu ozetinde ayrica toplam ve en son yil icin su toplamlari tutulur:

- Kariyer kosu sayisi, galibiyet sayisi ve ilk 3 sayisi
- Son yil kosu sayisi, galibiyet sayisi ve ilk 3 sayisi
- Secili jokey-at kombinasyonunun kosu ve galibiyet sayisi

Bu sayfada bulunan `X1` ve `X2` gibi diger sutunlar ham TJK tablosunda gorunse
de mevcut modelde alanlari olmadigi icin saklanmaz. Hava durumu ile erken/orta/gec
tempo indeksleri de bu akistan uretilmez; modeldeki ilgili alanlar `None` kalir.

#### Idman bilgileri (`IdmanIstatistikleri`)

`get_horse_statistics(entry, include_workouts=True)` varsayilaniyla, at idman
sayfasinda uygun tablo bulunursa su alanlar cekilir:

- Idman tarihi
- Idman hipodromu
- Pist
- Idman turu
- Idman jokeyi
- Durum
- Pistteki durum/siralama bilgisi (`P.Dur`)
- Detay
- Ilk parse edilebilen mesafe: 1400, 1200, 1000, 800, 600, 400 veya 200 metre
- Bu mesafeye ait derece, saniyeye cevrilmis olarak

TJK idman tablosunda birden fazla mesafe olsa da mevcut parser her satirda ilk
parse edilebilen mesafeyi tek bir `WorkoutRecord` olarak saklar; tum mesafe ve
derece kolonlarini ayri ayri korumaz. Idman sayfasi alinamazsa veya tablo yapisi
degisirse gecmis kosu verisi korunur, idman listesi bos kalir.

#### Gecmis yaris sonuclari (`GunlukYarisSonuclari`)

`get_daily_race_results()` ile secili tarih ve hipodrom icin su ozet cekilir:

- Yaris numarasi
- Start numarasi/at numarasi
- Bitis sirasi

Donus formati `{yaris_no: {at_no: bitis_sirasi}}` seklindedir. Sonuc sayfasinda
gorunen at adi, derece, ganyan, ikramiye ve diger sonuc sutunlari mevcut
sonuc parser'inda saklanmaz. CSV sonuc akisi da yalnizca `At No` satirlarindan
bitis sirasini uretir.

#### TJK'dan cekilmeyen veya bu adapter'da henuz bulunmayanlar

- Guvenilir, resmi ve sabit bir TJK REST API verisi
- JavaScript ile yuklenen her tablo ve AJAX yaniti
- Programdaki tum ham HTML/CSV sutunlarinin eksiksiz arsivi
- Atin ayrintili biyografisi, soy kutugu ve sahiplik gecmisi
- Jokey ve antrenorun bagimsiz kariyer istatistikleri
- Hava durumu ve pist kosullari
- Tum idman mesafelerinin ayni satirda ayri ayri dereceleri
- Sonuc sayfasinin tam odeme/ikramiye ve derece ayrintilari

Bu nedenle yukaridaki liste, TJK sitesinin teorik olarak gosterebildigi her
bilgiyi degil, bu projedeki mevcut parser'in gercekten modelleyip kullanabildigi
veri kapsamlarini ifade eder.

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

### TJK Kaynakli Feature'larin Anlami

Model artik sabit bir 32 feature listesini degil, TJK scraper'inin program,
gecmis kosu, kariyer ozeti, ganyan ve idman sayfalarindan elde ettigi sayisal
alanlari kullanir. Guncel liste `src/atyaris/ml/features.py` icindeki
`TJK_FEATURE_COLUMNS` sabitidir. Mevcut kosunun `finish_position`/`is_winner`
sonucu feature hesaplanirken kullanilmaz; gecmis form ve ozetler yalnizca
TJK'nin onceki kosularindan hesaplanir.

#### Yaris ve fiziksel bilgiler

| Feature | Anlami |
| --- | --- |
| `draw` | Atin start kulvari veya resmi cikis numarasi. |
| `weight` | Atin mevcut kosuda tasidigi agirlik, kilogram cinsinden. |
| `distance` | Yaris mesafesi, metre cinsinden. |
| `field_size` | Kosuya katilan toplam at sayisi. |

#### Gecmis form

Gecmis performans puani, bitis sirasi ve kosudaki at sayisi kullanilarak
normalize edilir. Birinci bitiren at yaklasik `1.0`, sonuncu bitiren at
`0.0` degerini alir.

| Feature | Anlami |
| --- | --- |
| `form_avg_3` | Son 3 kosudaki normalize edilmis performans ortalamasi. |
| `form_avg_5` | Son 5 kosudaki normalize edilmis performans ortalamasi. |
| `form_avg_10` | Son 10 kosudaki normalize edilmis performans ortalamasi. |
| `form_var_5` | Son 5 performansin degiskenligi; yuksek deger istikrarsizliga isaret eder. |
| `last_run_perf` | Atin son kosudaki normalize edilmis performansi. |
| `trend_3_10` | Kisa donem formu ile uzun donem formu farki: `form_avg_3 - form_avg_10`. |

#### Dinlenme, yorgunluk ve yaris sikligi

| Feature | Anlami |
| --- | --- |
| `days_since_last_race` | Son kosudan bu yana gecen gun sayisi. |
| `fatigue_score` | Dinlenme suresinin uzunlugunu sigmoid egriyle temsil eden skor. |
| `recovery_score` | Kisa dinlenme, yaris sikligi, sezonluk yuk ve son performansa gore toparlanma skoru. |
| `short_rest_flag` | Son kosudan itibaren 8 gunden az sure gectiyse `1`, aksi halde `0`. |
| `long_layoff_flag` | Son kosudan itibaren 75 gunden fazla sure gectiyse `1`, aksi halde `0`. |
| `race_frequency_3` | Son 3 kosunun ne kadar sik yapildigini temsil eden oran. |
| `race_frequency_5` | Son 5 kosunun ne kadar sik yapildigini temsil eden oran. |
| `seasonal_race_load` | Son 120 gundeki yaris sayisinin 120'ye bolunmesiyle elde edilen yuk. |

#### TJK gecmis kosu ve kariyer feature'lari

Bu feature'lar atin onceki kosularinda mevcut kosuya benzer sartlardaki
performansindan uretilir. Mesafe uyumunda yaklasik `+/- 200` metre araligi
kullanilir.

| Feature | Anlami |
| --- | --- |
| `distance_fit` | Mevcut mesafeye yakin kosulardaki performans uyumu. |
| `surface_fit` | Ayni pist turundeki, ornegin kum veya cim, performans uyumu. |
| `track_fit` | Ayni hipodromdaki gecmis performans uyumu. |

| `career_starts`, `career_wins`, `career_places` | TJK at ozeti kariyer toplamlaridir. |
| `last_year_starts`, `last_year_wins`, `last_year_places` | TJK at ozetindeki en son yil toplamlaridir. |
| `jockey_horse_combo_starts`, `jockey_horse_combo_wins` | TJK gecmis kosularindan secili jokey-at kombinasyonu toplamlaridir. |
| `history_avg_finish_position` | TJK gecmis kosularindaki ortalama bitis sirasidir. |
| `history_avg_field_size` | Gecmis kosulardaki ortalama at sayisidir. |
| `history_avg_weight` | Gecmis kosulardaki ortalama tasinan kilodur. |
| `history_avg_odds` | Gecmis kosulardaki ortalama TJK ganyanidir. |
| `history_avg_handicap_points` | Gecmis kosulardaki ortalama HP degeridir. |
| `history_avg_race_time_seconds` | Gecmis derecelerin saniye cinsinden ortalamasidir. |
| `history_avg_prize`, `history_avg_s20` | TJK gecmis satirlarindaki sayisal ikramiye ve S20 ortalamalaridir. |

#### TJK idman feature'lari

| Feature | Anlami |
| --- | --- |
| `workout_count` | TJK idman sayfasinda parse edilen idman satiri sayisidir. |
| `workout_avg_time_seconds` | Parse edilen idman derecelerinin ortalamasidir. |
| `workout_best_time_seconds` | Parse edilen idman derecelerinin en iyisidir. |
| `workout_avg_distance` | Parse edilen idman mesafelerinin ortalamasidir. |
| `days_since_last_workout` | Son TJK idmanindan bu yana gecen gun sayisidir. |

##### Eksik gecmis verinin yorumu

Gecmis kosusu bulunmayan bir at icin sistem egitimi durdurmaz. Bunun yerine
form feature'larinda genellikle `0.45`, dinlenme suresinde `30` gun, kosu
stillerinde her stil icin `0.25` ve uyum feature'larinda `0.45` oncul degerleri
kullanilir. Bu degerler gercek performans sonucu degil, bilinmeyen gecmisi
notr/ortalama kabul eden varsayilanlardir.

Gercek TJK egitim verisi daha once hesaplanmis feature'lari iceriyorsa bu
degerler tekrar hesaplanip varsayilanlarla ezilmez. Ancak at gecmisi TJK'den
alinamadiysa varsayilanlar kullanilir. Bu nedenle bir feature'in sayisal
olmasi, her satirda gercek gecmis verisi bulundugu anlamina gelmez.

`odds`, `market_probability_norm` ve `implied_probability` de feature setine
dahildir; bunlar TJK programindaki ganyandan uretilen piyasa feature'laridir.
TJK'nin yalnizca metin olarak verdigi at, jokey, antrenor, pist, hipodrom,
ekipman ve sinif adlari rastgele sayisal kodlara cevrilmez. Bu alanlar ham
metadata olarak korunur; aksi halde model kategoriler arasinda anlamsiz bir
buyukluk sirasi ogrenebilir.

### Modeller ve Egitim Sonrasi Degisiklikler Tablosu

Web arayuzundeki **Modeller ve egitim sonrasi degisiklikler** tablosu,
`model_runs` kayitlarini ve her modelin bir onceki modelden farklarini gosterir.
Degerlerin anlami su sekildedir:

| Baslik | Anlami |
| --- | --- |
| `Model` | Egitim kaydinin benzersiz model versiyonu, ornegin `model_v20260912_190835`. |
| `Egitim tarihi` | Model egitim kaydinin olusturuldugu tarih ve saat. |
| `Veri araligi` | Egitimde kayda alinan feature verisinin ilk ve son tarihi. |
| `Durum` | Modelin kayit durumu. `candidate` aday modeli, `production` kullanima alinmis modeli ifade eder. |
| `Egitim satiri` | Egitim feature tablosundaki toplam satir sayisi. Genellikle bir at-yaris kaydina karsilik gelir. |
| `Feature` | Model artifact'inin kullandigi guncel TJK kaynakli feature sayisidir. |
| `Holdout` | Zaman siralamasinin sonundan egitimden ayrilan test doneminin gun sayisi. Config'teki mevcut varsayilan `30` gundur. |
| `Kalibrasyon` | Ham model olasiliklarini gercek frekanslara uyarlamak icin kullanilan yontem: `isotonic`, `platt` veya `none`. |
| `Blend` | Kayit sirasinda stage-2 piyasa birlestirme modelinden alinan logistic agirlik/katsayi bilgisidir. Bu deger dogrudan yuzde olarak yorumlanmaz. |
| `Test log loss` | Holdout donemindeki olasilik tahminlerinin kaybi. Dusuk deger daha iyidir; yanlis ve asiri emin tahminleri daha fazla cezalandirir. |
| `Test Top-1` | Holdout yarislari icinde modelin birinci siraya koydugu atin gercekten kazandigi oran. Yuksek deger daha iyidir. |
| `Onceki model` | Bu kayittan hemen onceki egitim kaydinin model versiyonu. Ilk modelde `-` gorunur. |
| `Eklenen feature` | Onceki modele gore yeni modelde bulunan feature isimleri. |
| `Cikan feature` | Onceki modelde bulunup yeni modelden cikarilan feature isimleri. |
| `Kalibrasyon degisti` | Yeni modelin kalibrasyon yontemi onceki modelden farkliysa `Evet`, ayniysa `Hayir`. Ilk modelde `-` gorunur. |

#### Holdout ve metriklerin okunmasi

Egitim verisi tarih bazli bolunur. Eski donem model egitimi icin kullanilir;
kalibrasyon donemi olasiliklari duzeltir; son `holdout_days` gun ise modelin
daha once gormedigi test donemidir. Bu nedenle test metrikleri egitim
satirlarinda degil, holdout doneminde hesaplanir.

- **Test log loss:** Dusuk olmasi tercih edilir. Modelin dogru olasilik verip
  vermedigini, sadece dogru ati ilk siraya koyup koymadigini degil, olasiligin
  ne kadar emin verildigini de olcer.
- **Test Top-1:** Yuksek olmasi tercih edilir. Her yarista modelin en ust
  siraya koydugu atin kazanma oranidir.
- Iki metrik birlikte okunmalidir. Top-1 yukselirken log loss kotulesiyorsa
  model siralama olarak iyilesmis, fakat olasiliklerini fazla iddiali veya
  kotu kalibre edilmis olabilir.

`Eklenen feature`, `Cikan feature` ve `Kalibrasyon degisti` alanlari yalnizca
onceki egitim kaydiyla karsilastirmadir; tek basina yeni modelin daha iyi
oldugunu kanitlamaz. Model kalitesi icin holdout log loss, Top-1, kalibrasyon
egrisi ve walk-forward backtest birlikte incelenmelidir.

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

## HTML ML Tahmin Tablosunu Yorumlama

Web arayuzunde `Kaynak = ml (benter)` secildiginde kosu detay sayfasindaki
tablo su sirayla gorunur: `Rank`, `No`, `At`, `Ganyan`, `P(win)`, `P(2.)`,
`P(3.)`, `P(Top3)`, `Guven`, `Edge`, `EV`, `Kelly`, `Karar`. Sutun basliklarina
tiklanarak tablo ilgili degere gore siralanabilir. Siralama sadece gorunumu
degistirir; modelin kararini veya hesaplanan degerleri degistirmez.

### Kimlik ve piyasa sutunlari

- **Rank:** Modelin o kosudaki siralamasi. Dusuk rank daha ust sirayi ifade
  eder; tek basina bahis onerisi degildir.
- **No:** TJK start numarasi veya `draw` degeri. Modelin sirasi degil,
  programdaki atin numarasidir.
- **At:** At adi. Veri eslestirilemezse teknik `horse_id` gosterilebilir.
- **Ganyan:** Piyasanin sundugu ondalik oran. Ornegin `4.00`, kazanmasi
  halinde bir birim bahis icin toplam `4.00` birim donus anlamina gelir; net
  kar `4.00 - 1.00 = 3.00` birimdir. Oran yoksa piyasa temelli `Edge`, `EV`
  ve Kelly hesaplari guvenilir sekilde yapilamaz.

### Olasilik sutunlari

- **P(win):** Kalibrasyondan gecmis kazanma olasiligi. Ayni kosudaki atlar
  arasinda modelin kazanma sansini gosterir ve genellikle ilk siralama icin
  baslangic noktasi olmalidir. Kalibre edilmis olmasi, uzun vadede benzer
  olasilik verilen gruplarda gercek kazanma oraninin bu degerlere yakinlamasi
  amaclandigi anlamina gelir; garanti degildir.
- **P(2.):** Atin tam olarak ikinci bitirme olasiligi. P(win) ile ayni sey
  degildir; place olasiliklarini kazanma olasiligi yerine kullanmayin.
- **P(3.):** Atin tam olarak ucuncu bitirme olasiligi.
- **P(Top3):** Atin ilk uc icinde yer alma olasiligi. Yaklasik olarak
  `P(win) + P(2.) + P(3.)` toplamini temsil eder. Bu sutun plase/top-3
  karsilastirmasinda yararlidir; ancak mevcut `BET/NO_BET` karari kazanma
  olasiligi ve piyasa oranina gore verilir.

`P(2.)`, `P(3.)` ve `P(Top3)` kazanma olasiligindan Harville yaklasimiyla
turetilir. Bu nedenle bunlari ayri bir modelin bagimsiz kaniti gibi degil,
modelin kazanma siralamasindan turetilmis ek gorunumler olarak okuyun.

### Model-piyasa sutunlari

- **Guven:** Bu projede genel bir "kazanma garantisi" veya istatistiksel
  anlamda kalibrasyon guven araligi degildir. Model ile piyasanin arasindaki
  farkin buyuklugunden uretilen bir conviction proxy'sidir:
  `Guven = 0.5 + min(abs(Edge), 0.5)`. Bu nedenle piyasa ile daha fazla
  ayrisan tahmin daha yuksek Guven alabilir; ayrismanin dogru oldugunu garanti
  etmez.
- **Edge:** Model olasiligi ile normalize edilmis piyasa olasiligi arasindaki
  farktir: `Edge = P(win) - P(piyasa)`. Pozitif Edge, modelin piyasanin
  ima ettiginden daha yuksek kazanma sansi gordugu anlamina gelir. Ornegin
  `P(win) = 0.30` ve piyasa olasiligi `0.22` ise Edge `0.08` olur.
- **EV:** Bir birim bahis icin teorik beklenen net getiridir:
  `EV = P(win) * Ganyan - 1`. Pozitif EV, model varsayimlari dogru kabul
  edildiginde uzun vadeli deger adayi demektir; tek kosuda kazanma garantisi
  veya kesin kar anlamina gelmez. Ganyan `4.00`, P(win) `0.30` ise
  `EV = 0.30 * 4.00 - 1 = 0.20` olur.
- **Kelly:** Pozitif beklenen degerden hareketle onerilen bahis payidir.
  Once tam Kelly orani hesaplanir, sonra `ev_fractional_kelly` ile
  kucultulur ve `ev_max_kelly_fraction` ile sinirlanir. Tablodaki deger
  bakiye carpani olarak okunmalidir: `0.025`, yaklasik bakiyenin `%2.5`'i
  anlamina gelir. Bu bir zorunluluk degil, risk kontrollu bir ust sinirdir.
- **Karar:** Uygulamanin otomatik sinyalidir. `BET` icin ayni anda su
  kosullar saglanmalidir: P(win) minimum olasilik esigini gecmeli, Edge
  minimum Edge esigini gecmeli, EV minimum EV esigini gecmeli ve Kelly sifirdan
  buyuk olmalidir. Bunlardan biri bile saglanmazsa `NO_BET` verilir.

### Tabloyu kullanarak tahmin yapma sirasi

1. **Once kosunun model siralamasini okuyun:** Yuksek P(win) ve dusuk Rank,
   modelin kazanma adayi olarak gordugu atlari gosterir. Esit veya yakin
   P(win) degerlerinde P(Top3), Ganyan ve Guven ek karsilastirma saglar.
2. **Sonra piyasa ile karsilastirin:** Pozitif Edge, modelin piyasa oranindan
   daha iyimser oldugunu; negatif Edge, modelin piyasadan daha kotu gordugunu
   gosterir. Pozitif Edge tek basina yeterli degildir.
3. **EV ile deger kontrolu yapin:** P(win) yuksek olsa bile Ganyan cok dusukse
   EV negatif olabilir. Bu durumda at modelin favorisi olabilir ama bahis
   degeri olmayabilir.
4. **Kelly ile riski sinirlayin:** EV pozitif olsa bile Kelly cok kucukse
   farkin pratik onemi dusuk olabilir. Kelly, bahis yapilip yapilmayacagindan
   cok, yapilacaksa bakiye icindeki payin ne kadar olabilecegini gosterir.
5. **Son olarak Karar'i kontrol edin:** `BET`, sistem esiklerinin tamaminin
   gecildigi anlamina gelir; `NO_BET`, atin kesinlikle kazanamayacagi anlamina
   gelmez. Yalnizca mevcut filtrelerle yeterli istatistiksel/piyasa degeri
   bulunmadigini belirtir.

### Ornek okuma

Bir satirda `P(win) = 0.30`, `P(Top3) = 0.62`, `Ganyan = 4.00`, `Edge = 0.08`
ve `EV = 0.20` goruluyorsa model ati hem guclu bir kazanma adayi hem de
piyasanin ima ettiginden daha degerli goruyor demektir. Kelly `0.025` ise,
uygulamanin fractional Kelly ve ust limit ayarlarina gore teorik onerinin
bakiyenin yaklasik `%2.5`'i oldugu okunur. Buna ragmen `Karar = NO_BET` ise
minimum olasilik, Edge veya EV esiklerinden biri gecilmemis olabilir; tabloda
sadece gorunen yuvarlanmis degerler kullanildigi icin esik karsilastirmasini
arka plandaki tam sayilar yapar.

Bu tablo karar destek aracidir. Tek bir sutuna, tek bir kosuya veya yalnizca
`BET` etiketine bakarak kesin sonuc varsayilmamali; model performansi
walk-forward backtest, kalibrasyon ve gercek holdout sonuclariyla birlikte
degerlendirilmelidir.

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
