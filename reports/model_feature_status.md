# Model ve Feature Durum Ozeti

**Durum tarihi:** 2026-09-27
**Veri:** 7.252 etiketli at-yaris satiri, 41 tarih (2026-07-18..2026-09-16)
**Walk-forward:** 14 gun minimum egitim + 7 gun calibration, 27 OOS gun / 441 yaris
**Kalici sonuc:** Bu rapor; son izole reduced-feature evaluation adayi `%TEMP%` altindadir.

## Karar

Son reduced-feature aday health gate'ini gecmedi (`weight` isaret kontrolu) ve
`DO_NOT_PROMOTE` olarak tutuldu. Mevcut `models/phase1_logreg.joblib` dosyasi
calisma agacinda degismis ve eslik eden `health.json` `passed: false`; web
loader'i bu modeli reddediyor, yani su an ML tahmini sunulmuyor. Port 8000'deki
worker fail-closed davranisini gosterdi. Guncel UI status badge'i ve loglama
port 8001'de ayri worker ile dogrulandi. Tam reduced-feature health ve
walk-forward zinciri 2026-09-27'de ayri aday uzerinde tamamlandi; nested/LOO
kaniti yine `not_established`.

## Veri Duzeltmesi

TJK gecmis tablosundaki `St` kolonu at start/numara bilgisidir;
onceki parser bunu `field_size` diye kaydediyordu. Eski arsivdeki 146.707
finish/field-size ciftinin 56.287'sinde bitis sirasi alan boyutunu asiyordu.

- History arsivi: son kontrolde 147.215 satir. Yerel gunluk programla kesin
  eslesen 2.078 satirin gercek `field_size` degeri geri yuklendi; diger 145.137 deger eksik
  olarak duzeltildi. Geri yuklenen satirlarda imkansiz finish/field-size cifti
  kalmadi.
- Parser artik `St` degerini `field_size` yapmiyor. Bilinmeyen field size,
  hedef yarisin alan boyutuyla doldurulmuyor; form skoru yalnizca bilinen ve
  `finish_position <= field_size` olan gecmislerden uretiliyor.
- Training CSV'sindeki field-size kaynakli 14 feature 7.252 satirin tamaminda
  guvenilir etiketli yaris gecmisinden yeniden hesaplandi. `form_avg_5` araligi
  0..1; imkansiz pozisyon/alan boyutu satiri 0. `career_starts` pozitif olan
  6.850 satirdaki bagimsiz kariyer bilgisi korundu.

## Ayni Splitte Once / Sonra

Eski builder davranisi, ayni veri ve 14/7 split ile bellek icinde yeniden
calistirildi. Karsilastirma 27 gun ve 441 ortak yaris uzerindendir.

| Metrik | Eski builder davranisi | Duzeltilmis feature seti |
|---|---:|---:|
| Top-4 hit rate | 70.52% | 70.75% |
| Log-loss | 0.36605 | 0.36472 |
| Brier | 0.08820 | 0.08805 |
| Stage-1 / odds-implied korelasyonu | 0.22785 | 0.28392 |
| Final / odds-implied korelasyonu | 0.96969 | 0.97109 |
| Exact odds-order eslesmesi (269 yaris) | 35.32% | 29.00% |
| Odds Top-4 kumesinden farkli Top-4 | 34.94% | 35.69% |

Bu kucuk nokta tahmini degisimleri istatistiksel olarak anlamli iyilesme
gostermez. Onceki konusmada gecen `0.956`, Stage-1/market degil, final-model/
market korelasyonu olarak aktarilmisti. Eski JSON artefact'inin ustune
yazildigindan bu degerin eski run protokolu dogrulanamiyor. Ayni splitteki
Stage-1/odds korelasyonu eski davranista `0.22785` idi; `0.956` bu deger
degildir.

## Walk-Forward

| Odds dilimi | Yaris | At satiri | Eski Top-4 | Duzeltilmis Top-4 | Duzeltilmis log-loss |
|---|---:|---:|---:|---:|---:|
| Favorite | 195 | 882 | 96.41% | 96.92% | 0.59388 |
| Middle | 184 | 2.459 | 51.63% | 50.54% | 0.34739 |
| Longshot | 62 | 1.485 | 45.16% | 48.39% | 0.25732 |

Duzeltilmis tam model: Top-1 30.84%, Top-2 45.80%, Top-3 60.32%, Top-4
70.75%, log-loss 0.36472 ve Brier 0.08805. Baseline/enhanced tam walk-forward
Top-4 degerleri 70.75%; log-loss degerleri 0.3647194 ve 0.3647194'tur.

### Piyasa Bagimliligi ve Siralama

| Korelasyon / kontrol | Eski builder | Duzeltilmis |
|---|---:|---:|
| Stage-1 / market probability | 0.35361 | 0.43219 |
| Final / market probability | 0.69042 | 0.68357 |
| Stage-1 / odds-implied probability | 0.22785 | 0.28392 |
| Final / odds-implied probability | 0.96969 | 0.97109 |

Duzenlenmis `clean_races.csv` uzerinde 14/7 walk-forward bastan calistirilarak
degerler bagimsiz dogrulandi: Stage-1/market `0.432185`, final/market `0.683573`,
Stage-1/odds-implied `0.283923`, final/odds-implied `0.971086`. Stage-1'in
piyasa korelasyonu orta duzeyde; final olasilik piyasa ile daha uyumlu, fakat
`0.956` olarak olculmedi.

Duzeltilmis model 269 odds-uygun yarisin %29.00'unda tam odds siralamasiyla
ayni sirayi, %35.69'unda farkli Top-4 kumesini verdi. Formal divergence kontrolu
gecti; production gate ise nested performans kaniti olmadigi icin kapali.

Bu tablo ve yukaridaki feature comparison, field-size bagimliligi aktif listeden
cikarilmadan onceki walk-forward run'ina aittir. Guncel egitim secimi yalnizca
`handicap_points`, `draw`, `weight`, `class_drop_flag`,
`workout_sudden_improvement` ve `rest_optimal_fit` feature'larini kullanir.
`field_size` ve ondan turetilen form/context feature'lari, tarihsel kapsamin
%98.59'unda alan boyutu bilinmedigi icin aktif secimden cikarildi. Canli serving
artifact'i bu degisiklikle yeniden egitilmedi veya degistirilmedi.

## Nested ve Alpha Sonuclari

### Feature holdout + Bonferroni

41 validation yarisinda baseline ve enhanced Top-4 %65.45; longshot Top-4 her
ikisi icin %65.85. Log-loss 0.44718294 ve 0.44718290. Paired Top-4 farki ve CI
`[0, 0]`; log-loss farki pratikte sifir. Sonuc `not_established`.

### Alpha grid

Her alpha noktasi odds karsilastirmasina uygun 269 yaris; longshot orani 26
yaris uzerindendir.

| Stage-1 alpha | Top-4 | Longshot Top-4 | Log-loss |
|---:|---:|---:|---:|
| 0.0 | 77.32% | 3.85% | 0.88086 |
| 0.2 | 77.70% | 3.85% | 0.30425 |
| 0.4 | 77.70% | 11.54% | 0.29208 |
| 0.6 | 75.09% | 15.38% | 0.28936 |
| 1.0 | 58.74% | 23.08% | 0.30248 |

Alpha seciminde 18 gun/278 yaris kullanildi; secilen alpha 0.2. Bagimsiz
validation 9 gun ve 163 yaris; odds karsilastirmasina uygun yalnizca 45 yaris
ve 3 tarih vardi. Secilen alpha Top-4 %80.00, odds-sorted Top-4 %77.78;
Bonferroni CI `[0, 0.07143]`, gerekli minimum 30 tarih saglanmadi. Nihai model
icin nokta oranlari %84.44 ve %77.78 olsa da yalnizca 3 tarih oldugundan etiket
`not_established`. Alpha paired ve Bonferroni sonuc etiketleri de
`not_established`.

Validation alpha-0.2 karsilastirmasi 36 yarista iki tarafta da %75 longshot
Top-4 ve sifir paired fark verdi. Daha yuksek alpha icin kanit yok.

### En Yuksek Odds'li Placer

162 yaris / 13 validation tarihinde model exact match %25.31, market baseline
%19.14; fark +6.17 pp, Bonferroni CI `[0, 0.13980]`. Gereken minimum 30 tarih
olmadigi icin `not_established`.

### Leave-One-Out

Bes feature'in her biri 62 yaris uzerinde test edildi. Her LOO icin baseline ve
aday Top-4 %70.75; longshot Top-4 %48.39; paired Top-4 farki 0 ve paired
log-loss CI `[0, 0]`. Bes sonucun tamami `not_established`:

| Cikarilan feature | Sonuc |
|---|---|
| `jockey_change_upgrade` | `not_established` |
| `trainer_change_upgrade` | `not_established` |
| `class_drop_flag` | `not_established` |
| `workout_sudden_improvement` | `not_established` |
| `rest_optimal_fit` | `not_established` |

## Production Sagligi

### Log-Loss Sıçraması: Kök Neden

`1.55574` raporunu ureten aday artifact, acil rollback sirasinda bilerek
`7325f06` health-pass artifact'iyle degistirildi; aday binary artik mevcut
degil, dolayisiyla o artifact'in ham olasiliklari sonradan birebir tekrar
incelenemiyor. Ayni inference kodu ve guncel 110 yarislik holdout ile Git-HEAD
artifact'inde hata yeniden uretildi ve kontrollu karsilastirma kok nedeni
izole etti:

- Holdout: 1.484 satir / 110 yaris / 2026-09-11..2026-09-16; odds alaninin
  1.243 satiri (`%83.8`) `0` ve `odds_missing=1`.
- Ayni satirlarda `market_probability_norm` eksiksiz; her yaris icin toplami
  1.0. Stage-2 kodu `odds` varsa bunu bu dogru alandan once kullaniyordu; `odds=0`
  olan atlara 0 market olasiligi, dolayisiyla clipped logit `1e-8` veriliyordu.
- Stage-2 interaction bu sifir market logits'lerini cok buyuk utility
  farklarina tasiyordu. Ornek: Adana 2026-09-13/5'te kazanan `SAHLANTAY-1`
  icin Stage-1 `0.1076`, market p `0`, Stage-2 p `1.93e-22`; ayni yarisin
  en yuksek Stage-2 p'si `1.0` idi.
- Ayni artifact, ayni split, sadece dogru `market_probability_norm` kullanimi:
  eski satir-BCE `2.44648` -> `0.28963`; `<1e-4` olasilik sayisi `1.063` -> `0`,
  kazananlarda `<1e-4` `93` -> `0`. Stage-2 ham p araligi `2.66e-24..1.0`
  yerine `0.00115..0.72450` oldu. Bu kontrollu sonuc, split degisikliginden
  ziyade eksik-odds alaninin yanlis kullanilmasinin ana regression oldugunu
  gosteriyor.

Split de buyudu ve degisti: onceki health-pass raporu 5.940 satir/32 tarih ve
1.018 test satiri bildirirken, guncel bolumleme 7.252 satir/41 tarih ve 1.484
test satiri/110 yaris kapsiyor (test baslangici 2026-09-10'dan 2026-09-11'e
kaydi). Bu fark eski ve yeni tekil health sayilarinin birebir karsilastirilmasini
gecersiz kilar; fakat ayni guncel splitteki kontrollu odds-source deneyi asil
keskin olasilik bozulmasini aciklar.

Ikinci, bagimsiz raporlama kusuru da bulundu: onceki `_log_loss` her at satirini
bagimsiz binary hedef sayip tum satirlar uzerinden ortalama BCE aliyordu. Bu
kod tarihi raporlarda degismemisti, yani tek basina ani spike degil; fakat
yarista olasiliklar 1'e normalize edildiginden dogru birim bu degildir.
Fonksiyon artik her yaris icin kazanan at(lar)in toplam olasilik kutlesine
`-log` uygular; eski `0.23..0.25` degerleri yeni race-level log-loss ile
kiyaslanamaz.

Uygulanan Stage-2 duzeltmesinden sonra son bilinen health-pass artifact'i ayni
holdout'ta: race-level log-loss `2.26724`, market `2.36531`; Top-4 `%62.73` /
`%54.55`. Ham p araligi `0.00244..0.67589`; `<1e-4` veya `>0.9999` yok.
Bu yalnizca baseline tanisidir; artifact tam guncel health suite ile yeniden
degerlendirilmedi ve yeni aday/promosyon karari degildir.

### Uretim Artefact Durumu

Onceki asamada `7325f06` artifact'i geri yuklenmisti, ancak mevcut calisma
agacindaki model/health dosyalari o ciftten farkli ve Git'e commit edilmemis.
Guncel `models/phase1_logreg.joblib` SHA-256
`0b96e8eccb79c52e013a239007bd08881af3fa2ee8dd1fe9b91649c143d53a2b`; eslik eden
`models/phase1_logreg.health.json` `passed: false`, tek basarisiz check
`test_feature_signs` (weight katsayisi `+0.05001`, beklenen isaret negatif).
Dolayisiyla su an serving icin health-pass model yok. Port 8000'de calisan
Uvicorn worker `/predict?source=ml` istegini `ML modeli hazir degil` ile
reddetti; tahmin cikarmadi. Bu dosya cifti icin commit/versiyon etiketi yok,
dosyalar dirty working-tree icerigi.

Guncel kodla baslatilan ikinci worker [http://127.0.0.1:8001](http://127.0.0.1:8001)
prediction sayfasinda `passed: false`, artifact SHA-256 ve basarisiz kontrolu
gosterdi; ayni degerler `Model health status for /predict` INFO loguna yazildi.
8000 worker'i kullanici sureci oldugu icin yeniden baslatilmadi. Fail-closed
loader health JSON eksik/bozuk/`passed` false ise sessiz fallback yapmadan modeli
reddeder.

## Tahmin Snapshot Etkisi

Biriken prediction snapshot'inde 4 tarihe yayilmis 183 tutarsiz satir / 173 at
tespit edilmisti: 2025-04-30 (1), 2025-05-01 (60), 2026-09-26 (68),
2026-09-27 (54). Cache validator tutarsiz snapshot'i reddediyor; cache
invalidasyonu testleri gecti. Bu degerlendirme sirasinda TJK'dan canli snapshot
refetch yapilmadi; reddedilen eski satirlar model egitimine/backtest'e alinmadi.

### Eksik Tarihsel Field Size Kurtarma

Son arsiv taramasinda 147.215 history kaydinin 145.137'si (`%98.59`) halen
field size'siz; kayitlar 1.706 farkli tarihe (2020-06-15..2026-09-20)
yayiliyor. Yerel program arsiminde 46 tarih var; bunlar history tarihleriyle
44 gunde ortusuyor. Scraper resmi tarih/sehir gunluk CSV adresini olusturabiliyor,
ancak 2020-06-15 Bursa ve 2025-09-11 Istanbul ornekleri bos dondu. Bu nedenle
kalan 1.662 tarih icin tam refetch'in mumkun oldugu varsayilamaz. Tum tarih/sehir
kombinasyonlarini rate-limit ile yoklayan checkpoint'li bir backfill denenebilir;
resmi arsivde olmayan kayitlar icin tam kosucu listesini veren baska bir TJK
sonuc/arsiv kaynagi gerekir. `field_size` isteyen yeni career-place-rate gibi
feature'lar su an modele eklenmemeli; live history kapsami anlamli bir oran
elde edilmeden bu sinyaller `not_established`/experimental kalmalidir.

## Test Durumu

Tam `.venv\\Scripts\\python.exe -m pytest -q` suite'i `201 passed` ile
645.34 saniyede tamamlandi. Walk-forward temporal integrity testi tek basina
optimizasyon sonrasi `119.29` saniyede gecti; onceki profilsiz `229.10` saniyeye
kiyasla yaklasik %48 daha hizli. cProfile kosusu (profiling overhead ile
356.78 saniye) tekrarlanan per-race DataFrame siralamalarini hot path olarak
belirledi; bu siralamalar NumPy `lexsort` ile degistirildi.

Market siralama raporunda odds-implied baseline artik yalnizca decimal odds'tan
uretiliyor; `market_probability_used` ise gecerli normalized market feed'ini
kullanmaya devam ediyor. Bu iki kaynak ayrimi `%100` odds-sort baseline testini
dogru tanima baglar ve cached normalized olasiliklarin odds sentinel'larini
gecersiz kilmasini onler. Odds-implied kaynak ayrimi, field-size feature
seciminin ve serving artifact'inin yeniden egitilmesinin yerine gecmez.

## Tekrar Degerlendirme

Karar: izole reduced-feature adayini production'a tasima veya alpha agirligini
degistirme. Tam health/backtest zinciri calisti, ancak health `test_feature_signs`
weight katsayisi nedeniyle kaldi ve nested/Bonferroni sonuc `not_established`.
Serving artifact'i degistirilmedi; mevcut working-tree artifact'i da zaten
health-fail ve web tarafinda bloke ediliyor. Veri ufku 41 tarihle sinirli
oldugundan `DO_NOT_PROMOTE` korunmalidir.

## 2026-09-27 Log-Loss ve Reduced-Feature Dogrulamasi

### Sıçramanın Nedeni

Tarihsel 2026-09-17 sabit-holdout raporundaki `0.2582058`, 1.018 satir/92 yaris
uzerinde hesaplanan rowwise binary BCE'dir. Eski Git surumundeki
`model_health._log_loss` kodu bunu her at satiri icin
`-y log(p) - (1-y) log(1-p)` olarak ortaliyordu. Mevcut health metriği ise
normalize edilmis race olasiliklari icin yaris basina kazanan olasilik kutlesine
`-log(sum(p_winner))` uygular; dolayisiyla `0.26` ile `1.55` dogrudan
karsilastirilabilir degildir.

Ayni mevcut 1.484-satir/110-yaris health holdout'unda, yeni reduced aday icin
raw output rowwise BCE `0.29761`, race loss `2.29059`; kalibrasyon sonrasi
rowwise BCE `0.29309`, race loss `2.32904` oldu. Walk-forward'in satir-BCE
metriği `0.27574` cikti. Yani duzeltilmis modelde oldugu iddia edilen `1.55`
bandi olasilik davranisinin kendisi olarak yeniden uretilmiyor; bu sayilarin
bir bolumu farkli loss tanimindan geliyor. `1.55574` ureten eski aday binary
artik bulunmadigindan onun tam holdout raw output'u birebir kurtarilamaz.

Ayrica gercek, bagimsiz bir Stage-2 hata da kontrollu olarak yeniden uretildi.
Ayni mevcut artifact ve splitte normalized market kaynagi ile rowwise BCE
`0.28805`; eski odds-first extraction (odds `0` sentinel'larini market
olasiligi sayma) ile `2.27203` oldu. Eski yolda 1.066 satir `p < 1e-4`, 30
satir `p > 0.9999`; duzeltilmis yolda iki sayim da sifir. Race toplam olasilik
kutlesi her iki yolda da 1'e normalizeydi; bozulma, cok kucuk/1'e yakin at
olasiliklarinin dagilimiydi. Bu, onceki `1.55574` spike'inin olasilik tarafindaki
kok nedeni olan zero-odds sentinel/market-source uyumsuzlugunu aciklar.

Raw probability ornekleri, reduced aday, 2026-09-11..16 test split'i. Her
yarisin raw ve final calibrated toplam kutlesi `1.0`; listelenen atlar yarisin
tum satirlaridir:

| Yaris | At | Raw p | Kalibre p |
|---|---|---:|---:|
| Ankara-2026-09-15-3 | ANATOLIAN HERO-1 | 0.064463 | 0.092054 |
| Ankara-2026-09-15-3 | BEAUTIFUL LIGHT-2 | 0.054880 | 0.089284 |
| Ankara-2026-09-15-3 | YURIBOYKA-3 | 0.054614 | 0.089208 |
| Ankara-2026-09-15-3 | EMANET-4 | 0.048495 | 0.087482 |
| Ankara-2026-09-15-3 | KEN PARKER K-5 | 0.730500 | 0.554893 |
| Ankara-2026-09-15-3 | KEN PARKER-5 | 0.047048 | 0.087079 |
| İzmir-2026-09-12-9 | LUNGTA-1 | 0.053979 | 0.086983 |
| İzmir-2026-09-12-9 | HILL OF ANGELS K-2 | 0.761460 | 0.573665 |
| İzmir-2026-09-12-9 | HILL OF ANGELS-2 | 0.051821 | 0.086386 |
| İzmir-2026-09-12-9 | BELMONDO-3 | 0.045312 | 0.084609 |
| İzmir-2026-09-12-9 | PRINCESS ELİF-4 | 0.044826 | 0.084478 |
| İzmir-2026-09-12-9 | WIN LIKE A MAN-5 | 0.042601 | 0.083879 |

### Kalibrasyon Kontrolu

Kayitli calibrator `platt` LogisticRegression; reduced model ile ayni joblib
artifact'inde saklanir ve feature vektoru degil, modelin scalar raw
probability'sini alir. Katsayi `+3.60394`, intercept `-2.72804` (monotonik).
Calibration raw araligi `0.000085..0.942118`, test raw araligi
`0.000360..0.848288`; test satirlarinin `%100`'u calibration skor araliginda.
Final olasilik araligi `0.033941..0.573665`; asiri uc deger yok ve race
toplamlarindaki en buyuk hata `2.22e-16`. Kalibrator dagilimi bozup log-loss
sıçramasi yaratmiyor; race loss'u `2.29059`'dan `2.32904`'e hafif kotulestiriyor,
rowwise BCE'yi `0.29761`'den `0.29309`'a iyilestiriyor. Bu iki metriğin farkli
hedef birimler oldugunu yine gosteriyor.

### Reduced-Feature Tam Degerlendirme

Aday `%TEMP%\\horsewashing_reduced_feature_candidate_224358.joblib` altinda
egitildi; serving dosyalarina yazilmadi. Aktif feature'lar:
`handicap_points`, `draw`, `weight`, `class_drop_flag`,
`workout_sudden_improvement`, `rest_optimal_fit`. Health test log-loss'lari
(race-level raw): train `2.04185`, validation `1.99930`, test `2.29059`.
Health `passed: false`; tek failure `test_feature_signs`: weight katsayisi
`+0.07209`, beklenen isaret `-1`.

14/7 walk-forward, 27 OOS gun/441 yaris: Top-1 `%31.29`, Top-4 `%72.11`,
rowwise log-loss `0.27574`, Brier `0.07772`. Market korelasyonu:
Stage-1/market `0.46552`, final/market `0.97280`, Stage-1/odds `0.29989`,
final/odds `0.66803`. Odds-uygun 269 yarista model exact odds order `%34.20`;
market-feed ve odds-only baseline'lar odds order ile `%100` ayni. 172 yaris
gecersiz/eksik odds nedeniyle karsilastirma disinda. Model/market divergence
kontrolu `%90` esik altinda gecti; bu tek basina promote izni degil.

Nested holdout 41 yaris: baseline ve reduced aday Top-4 ikisi de `%66.82`,
race-log-loss ikisi de `0.28395`; paired fark `0`, Bonferroni CI sifir,
`not_established`. Placer nested karsilastirmasi 162 yaris/13 tarih:
model `%27.16`, market `%24.69`, Bonferroni CI `[-0.09091, 0.12937]`; gerekli
30 tarih saglanmadi, `not_established`. Altı aktif feature'in tamaminda LOO
sonucu `not_established`; LOO aile alpha'si `0.00833`. Market-ranking divergence
testi pass olsa da final karar `DO_NOT_PROMOTE` kalir.

Port 8001'deki status UI ve failed-health block odakli web testlerini gecti.
Son degisikliklerden sonra tam pytest suite `202 passed` ile `745.90` saniyede
tamamlandi.
