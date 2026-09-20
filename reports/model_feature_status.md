# Model ve Feature Durum Ozeti

**Durum tarihi:** 2026-09-20  
**Mevcut veri ufku:** Yaklasik 34 gun  
**Son tam walk-forward degerlendirme:** 333 yaris  
**Kalici sonuc dosyasi:** `models/phase1_feature_comparison.json`

## Kisa sonuc

Mevcut veri hacminde yeni feature'larin veya alpha seciminin performans katkisi
istatistiksel olarak kanitlanmadi. Bu sonuc feature'larin kotu oldugunu veya
koddan cikarilmalari gerektigini gostermez; longshot alt grubunda gozlem sayisi
cok dusuktur. Feature'lar ve alpha altyapisi aktif kodda korunur, fakat UI ve
raporlar sonucu `not_established` olarak etiketler.

## Test edilen hipotezler

### 1. Zaman sizintisi / point-in-time guvenligi

**Hipotez:** Bir yarisin veya idmanin gelecekteki sonucu, hedef yaris icin
feature uretiminde kullanilmiyor.

**Kontroller:**

- Hedef satirin `finish_position` ve `is_winner` degeri kendi feature'larini
  olusturmuyor.
- Gecmis yarislarda `race_date < target_race_date` filtresi uygulaniyor.
- Idmanlar hedef tarihten onceki kayitlarla sinirlaniyor.
- Lokal yeniden egitim temporal replay yolunu kullaniyor.
- Gelecekteki satirlar onceki feature degerlerini degistirmiyor.

**Sonuc:** Testler gecti. Bu hipotez icin mevcut kanit: `supported`.

### 2. Alpha / piyasa karisimi

**Hipotez:** Stage-1 model sinyali ile piyasa sinyalinin alpha/mix agirligi,
longshot Top-4 ve loss performansini guvenilir bicimde iyilestiriyor.

**Metodoloji:**

- Tarihlerin ilk %70'i alpha secim seti olarak kullanildi.
- Son %30 bagimsiz validation seti olarak tutuldu.
- Secim sonucu validation'da tekrar test edildi.
- Yarisa-paired bootstrap ve Bonferroni duzeltmesi uygulandi.

**Sonuc:** Onceki alpha deneylerinde secim setinde daha yuksek alpha secilebildi,
ancak bagimsiz validation ve Bonferroni araliklari sifiri dislamadi.
Etiket: `not_established`.

### 3. Bes yeni longshot feature

Test edilen feature'lar:

- `jockey_change_upgrade`
- `trainer_change_upgrade`
- `class_drop_flag`
- `workout_sudden_improvement`
- `rest_optimal_fit`

**Hipotez:** Bu sinyaller piyasanin fiyatlamadigi bilgi tasiyor ve longshot
Top-4 yakalamayi iyilestiriyor.

**Metodoloji:**

- Eski 8 feature'lik baseline ile yeni 13 feature'lik model ayni walk-forward
  fold'larinda calistirildi.
- Genel Top-4, log-loss ve odds dilimleri raporlandi.
- Longshot kazanan yarislarinda paired Top-4 karsilastirmasi yapildi.
- Nested hold-out: ilk %70 selection, son %30 bagimsiz validation.
- Nested paired testte Bonferroni duzeltmeli guven araligi kullanildi.
- Her yeni feature icin enhanced modelden o feature cikarilarak leave-one-out
  (LOO) testi yapildi.

## Son tam walk-forward sonucu

333 yarislik ortak testte:

| Metrik | Baseline | Yeni feature modeli |
|---|---:|---:|
| Genel Top-4 | 73.87% | 74.17% |
| Log-loss | 0.26121 | 0.26082 |
| Longshot Top-4 | 10.00% | 10.00% |
| Longshot yarisi | 30 | 30 |

Genel metriklerdeki kucuk farklar longshot tarafinda tekrarlanan bir kazanima
donusmedi.

### Feature-piyasa korelasyonlari

Korelasyonun dusuk olmasi sinyalin piyasa olasiligiyla ayni seyi tekrar
etmedigine dair olumlu bir isarettir; tek basina performans kaniti degildir.

| Feature | Piyasa korelasyonu |
|---|---:|
| `jockey_change_upgrade` | -0.0076 |
| `trainer_change_upgrade` | 0.0032 |
| `class_drop_flag` | 0.0000 |
| `workout_sudden_improvement` | 0.0057 |
| `rest_optimal_fit` | 0.0165 |

## Nested hold-out + Bonferroni sonucu

- Selection sonucu: `baseline`
- Bagimsiz validation longshot yarisi: 14
- Longshot paired Top-4 farki: `0.000`
- Bonferroni Top-4 CI: `[0.000, 0.000]`
- Paired log-loss iyilesmesi: `0.0000055`
- Bonferroni log-loss CI: `[-0.0000006, 0.0000122]`
- Nihai etiket: `not_established`

Bu nedenle yeni model icin `evidence_for_feature_model` etiketi uretmek icin
yeterli kanit yoktur.

## Leave-one-out katkisi

| Cikarilan feature | Longshot Top-4 farki | Log-loss iyilesmesi | Sonuc |
|---|---:|---:|---|
| `jockey_change_upgrade` | 0.000 | 0.0000958 | `not_established` |
| `trainer_change_upgrade` | 0.000 | 0.0000958 | `not_established` |
| `class_drop_flag` | 0.000 | 0.0000958 | `not_established` |
| `workout_sudden_improvement` | 0.000 | -0.0004300 | `not_established` |
| `rest_optimal_fit` | 0.000 | -0.0000135 | `not_established` |

LOO sonuclari bu veri hacminde hangi feature'in gurultu, hangisinin yararli
oldugunu guvenilir bicimde ayiramiyor. Idman feature'i cikarildiginda noktasal
log-loss farki biraz olumsuz gorunse de guven araligi sifiri kapsiyor.

## Neden feature'lar pasif sayiliyor?

Feature'lar koddan kaldirilmis degildir. Modelin secili feature listesinde ve
point-in-time feature builder'da tutulurlar. "Pasif" ifadesi su anlama gelir:

- Feature'lar hesaplanir ve model tarafindan kullanilabilir.
- Dusuk piyasa korelasyonu raporlanir.
- Ancak nested + Bonferroni testi kanit uretmedigi icin UI bunlari kanitlanmis
  avantaj olarak sunmaz.
- Veri buyudukce ayni otomatik degerlendirme yeniden calistirilir.

## Yeniden test tetikleyicisi

Asagidaki kosullardan biri saglandiginda degerlendirme yeniden yapilmalidir:

1. En az **200 longshot kazanan yaris** biriktiginde; mevcut 30 longshot
   yarislik test bunun cok altindadir.
2. Tercihen en az 12-18 aylik tarihsel veri ve her odds diliminde dengeli
   yarislar olustugunda.
3. Yeni veri eklendiginde once leakage testleri, sonra tam walk-forward,
   nested hold-out, Bonferroni ve LOO sirasi korunarak.

Yeni testte asgari rapor seti:

- Genel Top-4 ve log-loss
- Favorite/middle/longshot Top-4 ve log-loss
- Her yeni feature'in piyasa korelasyonu
- Longshot paired fark ve Bonferroni CI
- Nested selection/validation sonucu
- Feature bazli LOO tablosu
- `evidence_for_*` veya `not_established` etiketi

## Calisma notu

Bugunku karar: feature'lari cikarma, kanitlanmis iyilesme iddiasi da yapma.
Model ve alpha altyapisi veri buyudugunde otomatik yeniden test edilecek sekilde
korunuyor. UI'daki `not_established` etiketi bu durumla uyumludur.
