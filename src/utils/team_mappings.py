"""Static team-id mappings onto API-Football.

Contains FotMob and Svenska Spel → API-Football external ids.
"""

# FotMob map: data/fotmob_teams.csv → data/api_football_teams.csv
#   unresolved: data/unresolved_fotmob_team_mappings.csv
# Svenska Spel map: data/svenska_spel_teams.csv → data/api_football_teams.csv
#   unresolved: data/unresolved_svenska_spel_team_mappings.csv
# Prefer precision over coverage.

FOTMOB_TO_API_FOOTBALL_TEAMS = {
    1567: 227,  # Santa Clara -> Santa Clara
    1569: 7411,  # Kocaelispor -> Kocaelispor
    1634: 242,  # Famalicao -> Famalicao
    1693: 1165,  # KuPS -> KuPS
    1757: 125,  # América-MG -> America Mineiro
    1767: 1843,  # Havant and Waterlooville -> Havant & Wville
    1773: 260,  # OH Leuven -> OH Leuven
    1853: 584,  # Dundalk -> Dundalk
    1854: 3843,  # St. Patrick's Athletic -> St Patrick's Athl.
    1906: 9173,  # JäPS -> JäPS
    1925: 994,  # Göztepe -> Göztepe
    1933: 564,  # Başakşehir -> Başakşehir
    1965: 11675,  # Kristianstad FC -> Kristianstad
    1967: 6708,  # Vasalunds IF -> Vasalund
    2004: 376,  # Östersunds FK -> Ostersunds FK
    2006: 11671,  # Piteå -> Piteå
    2014: 2173,  # Norrby -> Norrby IF
    2159: 9187,  # KäPa -> KäPa
    2160: 2085,  # Klubi 04 -> Klubi-04
    2165: 272,  # KA Akureyri -> KA Akureyri
    2166: 1007,  # Rizespor -> Rizespor
    2260: 7644,  # Halesowen Town FC -> Halesowen Town
    2261: 4678,  # Billericay -> Billericay Town
    2305: 2143,  # KFUM -> KFUM Oslo
    2361: 2082,  # IF Gnistan -> Gnistan
    2488: 1830,  # Boreham Wood -> Boreham Wood
    4046: 7701,  # Bishop's Stortford -> Bishop's Stortford
    4047: 7690,  # Hornchurch -> AFC Hornchurch
    4081: 3573,  # Gaziantep FK -> Gaziantep FK
    4096: 1824,  # Guiseley -> Guiseley AFC
    4120: 1301,  # Rodez -> Rodez
    4130: 6766,  # Edinburgh City -> Edinburgh City
    4131: 652,  # Shamrock Rovers -> Shamrock Rovers
    4170: 1299,  # Boulogne -> Boulogne
    4438: 16596,  # Hammarby TFF -> Hammarby Talang
    4449: 2077,  # AC Oulu -> AC Oulu
    4451: 7726,  # Lancaster City -> Lancaster City
    4454: 1829,  # Dover -> Dover
    4472: 269,  # Keflavik -> Keflavik
    4493: 957,  # Levadiakos -> Levadiakos
    4594: 3840,  # Bohemian FC -> Bohemians
    4678: 996,  # Alanyaspor -> Alanyaspor
    4681: 3588,  # Eyüpspor -> Eyüpspor
    4685: 1004,  # Kasımpaşa -> Kasımpaşa
    4722: 6976,  # Egersund -> Egersund
    5751: 3854,  # Shelbourne -> Shelbourne
    5762: 7732,  # Merthyr Town -> Merthyr Town
    5763: 1831,  # Braintree Town -> Braintree
    5764: 8656,  # Leamington -> Leamington
    5767: 7665,  # Ramsgate -> Ramsgate
    6001: 1613,  # Columbus Crew -> Columbus Crew
    6002: 6763,  # Cove Rangers -> Cove Rangers
    6003: 9175,  # MP -> MP
    6017: 278,  # Vikingur Reykjavik -> Vikingur Reykjavik
    6042: 3845,  # Waterford FC -> Waterford
    6093: 8986,  # Walton and Hersham -> Walton & Hersham
    6095: 8657,  # Bath City -> Bath City
    6106: 863,  # Juve Stabia -> Juve Stabia
    6153: 6698,  # Skövde AIK -> Skövde AIK
    6160: 11674,  # IK Oddevold -> Oddevold
    6170: 6694,  # FC Rosengård -> Rosengård
    6173: 12657,  # Laholms FK -> Laholm
    6174: 6687,  # Lunds BK -> Lund
    6181: 2163,  # IFK Värnamo -> IFK Varnamo
    6183: 6702,  # Trollhättan FC -> Trollhättan
    6189: 1823,  # Gateshead FC -> Gateshead
    6194: 2241,  # Västerås SK -> Vasteras SK FK
    6195: 7757,  # Whitby -> Whitby Town
    6206: 7716,  # Folkestone Invicta -> Folkestone Invicta
    6238: 765,  # AFC Eskilstuna -> AFC Eskilstuna
    6241: 6697,  # Sandvikens IF -> Sandviken
    6258: 4677,  # Alfreton Town -> Alfreton Town
    6266: 506,  # Benevento -> Benevento
    6291: 2117,  # Fram Reykjavik -> Fram Reykjavik
    6305: 123,  # Sport Recife -> Sport Recife
    6314: 4687,  # Weston Super Mare -> Weston-super-Mare
    6315: 4688,  # AFC Telford United -> AFC Telford United
    6343: 275,  # Stjarnan -> Stjarnan
    6350: 1819,  # Barrow -> Barrow
    6355: 1297,  # Pau -> PAU
    6361: 3842,  # Sligo Rovers -> Sligo Rovers
    6379: 114,  # Paris FC -> Paris FC
    6390: 104,  # Red Star -> RED Star FC 93
    6397: 1607,  # Chicago Fire FC -> Chicago Fire
    6399: 1597,  # FC Dallas -> FC Dallas
    6413: 193,  # PEC Zwolle -> PEC Zwolle
    6414: 427,  # Telstar -> Telstar
    6422: 205,  # Fortuna Sittard -> Fortuna Sittard
    6432: 4685,  # Slough Town -> Slough Town
    6433: 410,  # Go Ahead Eagles -> GO Ahead Eagles
    6450: 4691,  # Gainsborough -> Gainsborough Trinity
    6479: 801,  # Pisa -> Pisa
    6488: 1581,  # Carrarese -> Carrarese
    6504: 1579,  # Monza -> Monza
    6514: 1602,  # Red Bull New York -> New York Red Bulls
    6544: 2172,  # Degerfors -> Degerfors IF
    6545: 812,  # Falkenbergs FF -> Falkenbergs FF
    6550: 1164,  # FC Inter Turku -> Inter Turku
    6563: 322,  # Ranheim -> Ranheim
    6580: 1609,  # New England Revolution -> New England Revolution
    6597: 650,  # VPS -> VPS
    6602: 1615,  # DC United -> DC United
    6603: 1596,  # San Jose Earthquakes -> San Jose Earthquakes
    6604: 1611,  # Sporting Kansas City -> Sporting Kansas City
    6606: 1606,  # Real Salt Lake -> Real Salt Lake
    6622: 2162,  # Åtvidaberg -> Atvidabergs FF
    6627: 4704,  # Eastbourne Borough -> Eastbourne Borough
    6633: 1168,  # TPS -> Turku PS
    6634: 587,  # IFK Mariehamn -> Mariehamn
    6637: 1605,  # LA Galaxy -> Los Angeles Galaxy
    6647: 8654,  # Kettering Town FC -> Kettering Town
    6661: 6781,  # Spartans FC -> Spartans
    6690: 2175,  # IK Brage -> IK brage
    6692: 2171,  # Varbergs BoIS FC -> Varbergs BoIS FC
    6694: 370,  # Sirius -> Sirius
    6701: 2076,  # FC KTP -> Kooteepee
    6702: 259,  # Lommel -> Lommel United
    6722: 528,  # Avellino -> Avellino
    7729: 140,  # Criciúma -> Criciuma
    7730: 1014,  # Lausanne -> Lausanne
    7732: 547,  # Girona -> Girona
    7733: 136,  # Vitória -> Vitoria
    7753: 1124,  # OFI Crete -> OFI
    7788: 420,  # Cambuur -> Cambuur
    7800: 997,  # Gençlerbirliği -> Gençlerbirliği S.K.
    7801: 520,  # Cremonese -> Cremonese
    7841: 226,  # Rio Ave -> Rio Ave
    7842: 230,  # Estoril -> Estoril
    7844: 224,  # Vitoria de Guimaraes -> Vitória SC
    7853: 433,  # Laval -> Laval
    7854: 537,  # Leganes -> Leganes
    7869: 713,  # Cordoba -> Cordoba
    7870: 2080,  # Haka -> Haka
    7876: 9580,  # Burgos CF -> Burgos
    7877: 118,  # Bahia -> Bahia
    7878: 715,  # Granada -> Granada CF
    7881: 517,  # Venezia -> Venezia
    7896: 606,  # Lugano -> FC Lugano
    7937: 2150,  # Hødd -> hodd
    7943: 488,  # Sassuolo -> Sassuolo
    7946: 1842,  # Harrogate Town -> Harrogate Town
    7954: 7715,  # Farnborough -> Farnborough
    7962: 8147,  # Chippenham Town -> Chippenham Town
    7978: 1393,  # Union St.Gilloise -> Union St. Gilloise
    7988: 330,  # Odds Ballklubb -> ODD Ballklubb
    7997: 813,  # Gefle -> Gefle IF
    8004: 827,  # IA Akranes -> IA Akranes
    8007: 326,  # Vålerenga -> Valerenga
    8009: 1821,  # Dagenham & Redbridge -> Dagenham & Redbridge
    8014: 372,  # Elfsborg -> IF Elfsborg
    8064: 274,  # Valur -> Valur Reykjavik
    8066: 903,  # Inverness Caledonian Thistle -> Inverness CT
    8071: 406,  # AGF -> Aarhus
    8113: 397,  # FC Midtjylland -> FC Midtjylland
    8119: 73,  # Rotherham United -> Rotherham
    8121: 77,  # Angers -> Angers
    8127: 2240,  # Mjällby -> Mjallby AIF
    8130: 271,  # KR Reykjavik -> KR Reykjavik
    8131: 1825,  # Maidstone United -> Maidstone Utd
    8133: 2146,  # Sandnes Ulf -> Sandnes ULF
    8141: 4669,  # Stirling Albion -> Stirling Albion
    8143: 6784,  # Stenhousemuir -> Stenhousemuir
    8145: 4250,  # Forfar Athletic -> Forfar Athletic
    8149: 182,  # Union Berlin -> Union Berlin
    8150: 191,  # Holstein Kiel -> Holstein Kiel
    8151: 270,  # FH Hafnarfjordur -> FH hafnarfjordur
    8152: 186,  # St. Pauli -> FC St. Pauli
    8158: 6777,  # Peterhead -> Peterhead
    8160: 1391,  # Alloa Athletic -> Alloa Athletic
    8165: 171,  # 1. FC Nürnberg -> 1. FC Nürnberg
    8175: 1369,  # Barnet -> Barnet
    8176: 4668,  # Airdrieonians -> Airdrie United
    8177: 159,  # Hertha BSC -> Hertha BSC
    8178: 168,  # Bayer Leverkusen -> Bayer Leverkusen
    8179: 2144,  # Kongsvinger -> Kongsvinger
    8180: 324,  # Strømsgodset -> Stromsgodset
    8188: 179,  # Magdeburg -> 1. FC Magdeburg
    8189: 2168,  # Ängelholms FF -> Angelholms FF
    8191: 44,  # Burnley -> Burnley
    8197: 46,  # Leicester City -> Leicester
    8203: 266,  # KV Mechelen -> KV Mechelen
    8226: 167,  # Hoffenheim -> 1899 Hoffenheim
    8232: 1660,  # Elversberg -> SV Elversberg
    8235: 6778,  # Queen's Park -> Queen's Park
    8239: 1123,  # Aris Thessaloniki -> Aris Thessalonikis
    8248: 363,  # Hammarby -> Hammarby FF
    8257: 4251,  # Montrose -> Montrose
    8259: 1600,  # Houston Dynamo FC -> Houston Dynamo
    8262: 181,  # Darmstadt -> SV Darmstadt 98
    8280: 4249,  # Arbroath -> Arbroath
    8282: 255,  # Livingston -> Livingston
    8283: 747,  # Barnsley -> Barnsley
    8284: 253,  # Dundee FC -> Dundee
    8287: 154,  # Fortaleza -> Fortaleza EC
    8288: 9692,  # Eldense -> Eldense
    8295: 785,  # Karlsruher SC -> Karlsruher SC
    8297: 2170,  # GAIS -> Gais
    8299: 268,  # IBV Vestmannaeyjar -> IBV Vestmannaeyjar
    8302: 536,  # Sevilla -> Sevilla
    8305: 546,  # Getafe -> Getafe
    8306: 534,  # Las Palmas -> Las Palmas
    8310: 766,  # Halmstads BK -> Halmstad
    8311: 99,  # Clermont Foot -> Clermont Foot
    8313: 1381,  # Tranmere Rovers -> Tranmere
    8314: 1610,  # Colorado Rapids -> Colorado Rapids
    8315: 531,  # Athletic Club -> Athletic Club
    8321: 1382,  # Dumbarton -> Dumbarton
    8332: 276,  # Breidablik -> Breidablik
    8333: 369,  # Trelleborgs FF -> Trelleborg
    8338: 670,  # Derry City -> Derry City
    8339: 3850,  # Drogheda United -> Drogheda United
    8342: 569,  # Club Brugge -> Club Brugge KV
    8344: 43,  # Cardiff City -> Cardiff
    8345: 1836,  # Woking -> Woking
    8346: 1359,  # Luton Town -> Luton
    8348: 215,  # Moreirense -> Moreirense
    8349: 377,  # AIK -> AIK Stockholm
    8350: 745,  # Kaiserslautern -> 1. FC Kaiserslautern
    8351: 1373,  # Leyton Orient -> Leyton Orient
    8357: 178,  # Greuther Fürth -> SpVgg Greuther Fürth
    8358: 160,  # Freiburg -> SC Freiburg
    8359: 373,  # GIF Sundsvall -> GIF Sundsvall
    8370: 728,  # Rayo Vallecano -> Rayo Vallecano
    8371: 727,  # Osasuna -> Osasuna
    8372: 545,  # Eibar -> Eibar
    8385: 724,  # Cadiz -> Cadiz
    8391: 400,  # FC København -> FC Copenhagen
    8393: 722,  # Albacete -> Albacete
    8401: 1357,  # Plymouth Argyle -> Plymouth
    8402: 327,  # Bodø/Glimt -> Bodo/Glimt
    8404: 757,  # Aalesund -> Aalesund
    8405: 7012,  # Moss -> Moss
    8406: 170,  # Augsburg -> FC Augsburg
    8409: 6762,  # Clyde -> Clyde
    8410: 401,  # Randers FC -> Randers FC
    8411: 59,  # Preston North End -> Preston
    8412: 1340,  # Scunthorpe United -> Scunthorpe
    8414: 405,  # OB -> Odense
    8415: 2073,  # Silkeborg -> Silkeborg
    8416: 1361,  # Colchester United -> Colchester
    8417: 2149,  # Fredrikstad -> Fredrikstad
    8422: 331,  # Rosenborg -> Rosenborg
    8424: 1166,  # FC Lahti -> Lahti
    8425: 12578,  # Enköping -> Enköping
    8426: 901,  # Partick Thistle -> Partick
    8427: 56,  # Bristol City -> Bristol City
    8428: 367,  # Häcken -> BK Hacken
    8429: 248,  # Hamilton Academical -> Hamilton Academical
    8430: 1379,  # Lincoln City -> Lincoln
    8448: 2159,  # Hamarkameratene -> Ham-Kam
    8449: 378,  # IFK Norrköping -> IFK Norrkoping
    8451: 1335,  # Charlton Athletic -> Charlton
    8455: 49,  # Chelsea -> Chelsea
    8456: 50,  # Manchester City -> Manchester City
    8457: 1388,  # Dunfermline Athletic -> Dunfermline
    8460: 185,  # Paderborn -> SC Paderborn 07
    8462: 1355,  # Portsmouth -> Portsmouth
    8463: 63,  # Leeds United -> Leeds
    8464: 413,  # NEC Nijmegen -> NEC Nijmegen
    8465: 1818,  # Aldershot Town -> Aldershot Town
    8466: 41,  # Southampton -> Southampton
    8467: 258,  # St. Johnstone -> ST Johnstone
    8468: 319,  # Brann -> Brann
    8472: 746,  # Sunderland -> Sunderland
    8473: 2075,  # FF Jaro -> FF Jaro
    8476: 321,  # Lillestrøm -> Lillestrom
    8478: 759,  # Viking -> Viking
    8480: 183,  # Dynamo Dresden -> Dynamo Dresden
    8481: 102,  # Nancy -> Nancy
    8483: 1356,  # Blackpool -> Blackpool
    8484: 1343,  # Bradford City -> Bradford
    8485: 252,  # Aberdeen -> Aberdeen
    8487: 396,  # Sønderjyske -> Sonderjyske
    8488: 1366,  # Hartlepool United -> Hartlepool
    8489: 1375,  # Morecambe -> Morecambe
    8493: 1339,  # Rochdale -> Rochdale
    8500: 2165,  # Ljungskile -> Ljungskile SK
    8501: 371,  # Brommapojkarna -> IF Brommapojkarna
    8509: 333,  # Sarpsborg 08 -> Sarpsborg 08 FF
    8510: 764,  # Jönköping S. -> Jonkopings Sodra
    8511: 2176,  # Landskrona BoIS -> Landskrona BoIS
    8512: 328,  # FK Haugesund -> Haugesund
    8514: 128,  # Santos FC -> Santos
    8517: 120,  # Botafogo RJ -> Botafogo
    8521: 106,  # Brest -> Stade Brestois 29
    8522: 507,  # Ascoli -> Ascoli
    8524: 499,  # Atalanta -> Atalanta
    8525: 195,  # Willem II -> Willem II
    8527: 365,  # Örebro -> Orebro SK
    8528: 61,  # Wigan Athletic -> Wigan
    8529: 490,  # Cagliari -> Cagliari
    8531: 2142,  # Bryne -> Bryne
    8534: 511,  # Empoli -> Empoli
    8535: 502,  # Fiorentina -> Fiorentina
    8540: 522,  # Palermo -> Palermo
    8543: 487,  # Lazio -> Lazio
    8548: 257,  # Rangers -> Rangers
    8549: 70,  # Middlesbrough -> Middlesbrough
    8550: 112,  # Metz -> Metz
    8558: 540,  # Espanyol -> Espanyol
    8559: 68,  # Bolton Wanderers -> Bolton
    8560: 548,  # Real Sociedad -> Real Sociedad
    8563: 575,  # AEK Athens -> AEK Athens FC
    8564: 489,  # Milan -> AC Milan
    8571: 734,  # Kortrijk -> Kortrijk
    8581: 539,  # Levante -> Levante
    8583: 108,  # Auxerre -> Auxerre
    8586: 47,  # Tottenham Hotspur -> Tottenham
    8588: 116,  # Lens -> Lens
    8592: 81,  # Marseille -> Marseille
    8593: 194,  # Ajax -> Ajax
    8595: 407,  # Brøndby IF -> Brondby
    8596: 1389,  # Falkirk -> Falkirk
    8597: 250,  # Kilmarnock -> Kilmarnock
    8598: 10157,  # Darlington -> Darlington 1883
    8599: 1380,  # Macclesfield FC -> Macclesfield
    8600: 494,  # Udinese -> Udinese
    8601: 6705,  # Umeå -> Umeå FC
    8602: 39,  # Wolverhampton Wanderers -> Wolves
    8603: 543,  # Real Betis -> Real Betis
    8605: 320,  # Kristiansund -> Kristiansund BK
    8608: 325,  # Tromsø -> Tromso
    8609: 332,  # Sandefjord -> Sandefjord
    8611: 415,  # FC Twente -> Twente
    8614: 426,  # Sparta Rotterdam -> Sparta Rotterdam
    8616: 758,  # Sogndal -> Sogndal
    8619: 619,  # PAOK Thessaloniki -> PAOK
    8622: 607,  # Konyaspor -> Konyaspor
    8633: 541,  # Real Madrid -> Real Madrid
    8634: 529,  # Barcelona -> Barcelona
    8635: 554,  # Anderlecht -> Anderlecht
    8636: 505,  # Inter -> Inter
    8637: 645,  # Galatasaray -> Galatasaray
    8638: 553,  # Olympiacos -> Olympiakos Piraeus
    8639: 79,  # Lille -> Lille
    8640: 197,  # PSV Eindhoven -> PSV Eindhoven
    8641: 2174,  # Östers IF -> Osters IF
    8645: 1348,  # Milton Keynes Dons -> Milton Keynes Dons
    8646: 8146,  # Boston United -> Boston United
    8647: 1362,  # Crawley Town -> Crawley Town
    8648: 1383,  # Greenock Morton -> Morton
    8649: 902,  # Ross County -> Ross County
    8650: 40,  # Liverpool -> Liverpool
    8651: 1337,  # Northampton Town -> Northampton
    8652: 1341,  # Southend United -> Southend
    8653: 1338,  # Oxford United -> Oxford United
    8654: 48,  # West Ham United -> West Ham
    8655: 67,  # Blackburn Rovers -> Blackburn
    8657: 62,  # Sheffield United -> Sheffield Utd
    8658: 54,  # Birmingham City -> Birmingham
    8659: 60,  # West Bromwich Albion -> West Brom
    8660: 7728,  # Lewes -> Lewes
    8661: 798,  # Mallorca -> Mallorca
    8667: 64,  # Hull City -> Hull City
    8668: 45,  # Everton -> Everton
    8669: 1346,  # Coventry City -> Coventry
    8670: 718,  # Real Oviedo -> Oviedo
    8671: 1360,  # Accrington Stanley -> Accrington ST
    8674: 202,  # FC Groningen -> Groningen
    8676: 1358,  # Wycombe Wanderers -> Wycombe
    8677: 1350,  # Peterborough United -> Peterborough
    8678: 35,  # AFC Bournemouth -> Bournemouth
    8679: 8664,  # Welling United -> Welling United
    8680: 1372,  # Cheltenham Town -> Cheltenham
    8682: 1298,  # Le Mans -> Le Mans
    8686: 497,  # Roma -> AS Roma
    8689: 97,  # Lorient -> Lorient
    8695: 611,  # Fenerbahçe -> Fenerbahçe
    8696: 4665,  # Racing Santander -> Racing Santander
    8697: 162,  # Werder Bremen -> Werder Bremen
    8702: 119,  # Internacional -> Internacional
    8717: 7616,  # Bedford Town -> Bedford Town
    8721: 161,  # Wolfsburg -> VfL Wolfsburg
    8722: 192,  # 1. FC Köln -> 1. FC Köln
    9746: 111,  # Le Havre -> Le Havre
    9747: 90,  # Guingamp -> Guingamp
    9748: 80,  # Lyon -> Lyon
    9750: 3603,  # Samsunspor -> Samsunspor
    9752: 998,  # Trabzonspor -> Trabzonspor
    9758: 6764,  # East Fife -> East Fife
    9764: 762,  # Gil Vicente -> GIL Vicente
    9767: 147,  # Coritiba -> Coritiba
    9768: 228,  # Sporting CP -> Sporting CP
    9769: 130,  # Grêmio -> Gremio
    9770: 127,  # Flamengo -> Flamengo
    9772: 211,  # Benfica -> Benfica
    9773: 212,  # FC Porto -> FC Porto
    9775: 1324,  # VfL Osnabrück -> VfL Osnabrück
    9776: 744,  # Eintracht Braunschweig -> Eintracht Braunschweig
    9777: 2184,  # Servette -> Servette FC
    9780: 4724,  # Alverca -> Alverca
    9781: 135,  # Cruzeiro -> Cruzeiro
    9783: 544,  # Deportivo A Coruña -> Deportivo La Coruna
    9784: 1363,  # Crewe Alexandra -> Crewe
    9785: 1349,  # Oldham Athletic -> Oldham
    9786: 1345,  # Chesterfield -> Chesterfield
    9788: 163,  # Borussia Mönchengladbach -> Borussia Mönchengladbach
    9789: 165,  # Borussia Dortmund -> Borussia Dortmund
    9790: 175,  # Hamburger SV -> Hamburger SV
    9792: 748,  # Burton Albion -> Burton Albion
    9794: 1840,  # Ebbsfleet United -> Ebbsfleet United
    9795: 1353,  # Swindon Town -> Swindon Town
    9796: 37,  # Huddersfield Town -> Huddersfield
    9797: 1820,  # Chester FC -> Chester
    9798: 53,  # Reading -> Reading
    9799: 1351,  # Port Vale -> Port Vale
    9800: 251,  # St. Mirren -> ST Mirren
    9802: 364,  # Djurgården -> Djurgardens IF
    9804: 503,  # Torino -> Torino
    9808: 131,  # Corinthians -> Corinthians
    9810: 169,  # Eintracht Frankfurt -> Eintracht Frankfurt
    9812: 2153,  # Raufoss -> Raufoss
    9814: 404,  # AC Horsens -> AC Horsens
    9817: 38,  # Watford -> Watford
    9818: 1374,  # Mansfield Town -> Mansfield Town
    9819: 1376,  # Notts County -> Notts County
    9823: 157,  # Bayern München -> Bayern München
    9825: 42,  # Arsenal -> Arsenal
    9826: 52,  # Crystal Palace -> Crystal Palace
    9828: 1378,  # Forest Green Rovers -> Forest Green
    9829: 91,  # Monaco -> Monaco
    9830: 83,  # Nantes -> Nantes
    9831: 84,  # Nice -> Nice
    9833: 1364,  # Exeter City -> Exeter City
    9834: 1370,  # Cambridge United -> Cambridge United
    9836: 89,  # Dijon -> Dijon
    9837: 93,  # Reims -> Reims
    9841: 1837,  # Wrexham -> Wrexham
    9847: 85,  # Paris Saint-Germain -> Paris Saint Germain
    9848: 95,  # Strasbourg -> Strasbourg
    9850: 71,  # Norwich City -> Norwich
    9851: 94,  # Rennes -> Rennes
    9853: 1063,  # Saint-Etienne -> Saint Etienne
    9855: 101,  # Grenoble -> Grenoble
    9857: 500,  # Bologna -> Bologna
    9859: 811,  # Helsingborg -> Helsingborg
    9860: 254,  # Heart of Midlothian -> Heart Of Midlothian
    9861: 649,  # HJK -> HJK Helsinki
    9862: 151,  # Goiás -> Goias
    9863: 124,  # Fluminense -> Fluminense
    9864: 535,  # Malaga -> Malaga
    9865: 723,  # Almeria -> Almeria
    9866: 542,  # Deportivo Alaves -> Alaves
    9867: 719,  # Tenerife -> Tenerife
    9869: 731,  # Sporting Gijon -> Sporting Gijon
    9874: 115,  # Sochaux -> Sochaux
    9875: 492,  # Napoli -> Napoli
    9876: 504,  # Hellas Verona -> Hellas Verona
    9879: 36,  # Fulham -> Fulham
    9882: 498,  # Sampdoria -> Sampdoria
    9885: 496,  # Juventus -> Juventus
    9887: 899,  # Modena -> Modena
    9888: 867,  # Lecce -> Lecce
    9889: 1693,  # Mantova -> Mantova
    9891: 512,  # Frosinone -> Frosinone
    9892: 374,  # Kalmar FF -> Kalmar FF
    9893: 366,  # IFK Göteborg -> IFK Goteborg
    9894: 7266,  # Worksop Town -> Worksop Town
    9895: 1344,  # Bury -> Bury
    9896: 1352,  # Shrewsbury Town -> Shrewsbury
    9902: 57,  # Ipswich Town -> Ipswich
    9903: 1354,  # Doncaster Rovers -> Doncaster
    9904: 166,  # Hannover 96 -> Hannover 96
    9905: 164,  # Mainz 05 -> FSV Mainz 05
    9906: 530,  # Atletico Madrid -> Atletico Madrid
    9907: 625,  # Lyngby -> Lyngby
    9908: 207,  # FC Utrecht -> Utrecht
    9910: 538,  # Celta Vigo -> Celta Vigo
    9911: 176,  # Bochum -> VfL Bochum
    9912: 188,  # Arminia Bielefeld -> Arminia Bielefeld
    9913: 1387,  # Ayr United -> Ayr Utd
    9914: 6767,  # Elgin City -> Elgin City
    9915: 4700,  # Altrincham -> Altrincham
    9916: 1828,  # York City -> York
    9917: 329,  # Molde -> Molde
    9918: 323,  # Stabæk -> Stabaek
    9919: 334,  # Start -> Start
    9923: 7721,  # Hednesford -> Hednesford Town
    9924: 1384,  # Queen of the South -> Queen of the South
    9925: 247,  # Celtic -> Celtic
    9926: 6785,  # Stranraer -> Stranraer
    9927: 256,  # Motherwell -> Motherwell
    9931: 551,  # Basel -> FC Basel 1893
    9937: 55,  # Brentford -> Brentford
    9938: 1386,  # Dundee United -> Dundee Utd
    9939: 2070,  # Viborg -> Viborg
    9941: 96,  # Toulouse -> Toulouse
    9956: 1013,  # Grasshopper -> Grasshoppers
    9984: 741,  # Cercle Brugge -> Cercle Brugge
    9985: 733,  # Standard Liege -> Standard Liege
    9986: 736,  # Sporting Charleroi -> Charleroi
    9987: 742,  # Genk -> Genk
    9988: 740,  # Royal Antwerp -> Antwerp
    9991: 631,  # Gent -> Gent
    9997: 735,  # St.Truiden -> St. Truiden
    10000: 600,  # Zulte Waregem -> Zulte Waregem
    10001: 261,  # Westerlo -> KVC Westerlo
    10002: 2166,  # Örgryte -> Orgryte IS
    10003: 76,  # Swansea City -> Swansea
    10004: 58,  # Millwall -> Millwall
    10005: 1365,  # Grimsby Town -> Grimsby
    10006: 1342,  # Walsall -> Walsall
    10007: 4686,  # Stockport County -> Stockport County
    10021: 2148,  # Åsane -> Asane
    10104: 1334,  # Bristol Rovers -> Bristol Rovers
    10163: 74,  # Sheffield Wednesday -> Sheffield Wednesday
    10167: 523,  # Parma -> Parma
    10168: 1687,  # Catanzaro -> Catanzaro
    10170: 69,  # Derby County -> Derby
    10171: 895,  # Como -> Como
    10172: 72,  # Queens Park Rangers -> QPR
    10173: 1347,  # Gillingham -> Gillingham
    10174: 4708,  # St Albans -> St Albans City
    10179: 630,  # Sion -> FC Sion
    10187: 12260,  # Atromitos -> Atromitos
    10188: 549,  # Beşiktaş -> Beşiktaş
    10189: 174,  # Schalke 04 -> FC Schalke 04
    10190: 1011,  # St. Gallen -> FC ST. Gallen
    10191: 1012,  # Thun -> FC Thun
    10192: 565,  # Young Boys -> BSC Young Boys
    10193: 1827,  # Torquay United -> Torquay
    10194: 75,  # Stoke City -> Stoke City
    10195: 1841,  # FC Halifax Town -> FC Halifax Town
    10196: 1371,  # Carlisle United -> Carlisle
    10197: 1826,  # Southport -> Southport
    10198: 1377,  # Yeovil Town -> Yeovil Town
    10199: 644,  # Luzern -> FC Luzern
    10200: 617,  # Panathinaikos -> Panathinaikos
    10202: 398,  # Nordsjælland -> FC Nordsjaelland
    10203: 65,  # Nottingham Forest -> Nottingham Forest
    10204: 51,  # Brighton & Hove Albion -> Brighton
    10205: 533,  # Villarreal -> Villarreal
    10212: 214,  # Maritimo -> Maritimo
    10214: 225,  # Nacional -> Nacional
    10217: 198,  # ADO Den Haag -> ADO Den Haag
    10218: 196,  # Excelsior -> Excelsior
    10224: 7006,  # Lyn -> Lyn
    10225: 2169,  # Assyriska FF -> Assyriska FF
    10228: 210,  # SC Heerenveen -> Heerenveen
    10229: 201,  # AZ Alkmaar -> AZ Alkmaar
    10233: 495,  # Genoa -> Genoa
    10235: 209,  # Feyenoord -> Feyenoord
    10237: 375,  # Malmö FF -> Malmo FF
    10242: 110,  # Troyes -> Estac Troyes
    10243: 783,  # FC Zürich -> FC Zurich
    10249: 82,  # Montpellier -> Montpellier
    10250: 1385,  # Raith Rovers -> Raith Rovers
    10251: 249,  # Hibernian -> Hibernian
    10252: 66,  # Aston Villa -> Aston Villa
    10253: 1368,  # Stevenage -> Stevenage
    10254: 7752,  # Tamworth -> Tamworth
    10260: 33,  # Manchester United -> Manchester United
    10261: 34,  # Newcastle United -> Newcastle
    10262: 1367,  # Newport County -> Newport County
    10264: 217,  # Braga -> SC Braga
    10267: 532,  # Valencia -> Valencia
    10268: 797,  # Elche -> Elche
    10269: 172,  # VfB Stuttgart -> VfB Stuttgart
    10272: 1062,  # Atlético-MG -> Atletico-MG
    10273: 134,  # Athletico Paranaense -> Atletico Paranaense
    10274: 152,  # Juventude -> Juventude
    10276: 133,  # Vasco da Gama -> Vasco DA Gama
    10277: 126,  # São Paulo -> Sao Paulo
    10279: 5254,  # Castellon -> Castellón
    10281: 720,  # Real Valladolid -> Valladolid
    10283: 121,  # Palmeiras -> Palmeiras
    10284: 4695,  # Kidderminster Harriers -> Kidderminster Harriers
    45722: 4696,  # Leatherhead -> Leatherhead
    45723: 1336,  # Fleetwood Town -> Fleetwood Town
    45724: 4689,  # Chelmsford -> Chelmsford City
    45725: 1838,  # Maidenhead United -> Maidenhead
    45729: 1832,  # Bromley -> Bromley
    45731: 8655,  # King's Lynn Town -> King's Lynn Town
    45807: 7615,  # Basingstoke -> Basingstoke Town
    46036: 6757,  # Annan Athletic -> Annan Athletic
    47214: 1304,  # Dunkerque -> Dunkerque
    56453: 1601,  # Toronto FC -> Toronto FC
    73158: 6706,  # Utsiktens BK -> Utsikten
    80654: 955,  # Asteras Tripolis -> Asteras Tripolis
    94937: 180,  # FC Heidenheim -> 1. FC Heidenheim
    104822: 145,  # Avaí -> Avai
    105552: 4680,  # Hampton & Richmond -> Hampton & Richmond
    105553: 7265,  # Workington -> Workington
    105554: 7723,  # Horsham -> Horsham
    109705: 794,  # Bragantino -> RB Bragantino
    111120: 6699,  # Sollentuna FK -> Sollentuna
    129937: 8903,  # Quorn -> Quorn
    130394: 1595,  # Seattle Sounders FC -> Seattle Sounders
    158085: 240,  # Arouca -> Arouca
    158316: 1835,  # Sutton United -> Sutton Utd
    158317: 7640,  # Evesham United -> Evesham United
    158319: 1333,  # AFC Wimbledon -> AFC Wimbledon
    158320: 7622,  # Bury Town -> Bury Town
    158321: 4701,  # Brackley Town -> Brackley Town
    158390: 4703,  # Curzon -> Curzon Ashton
    158543: 7727,  # Leiston -> Leiston
    161195: 1614,  # CF Montreal -> CF Montreal
    161744: 9585,  # Real Sociedad B -> Real Sociedad II
    161800: 7741,  # Redditch United -> Redditch United
    161801: 1834,  # Solihull Moors -> Solihull Moors
    161802: 4694,  # Hyde United -> Hyde United
    161803: 1822,  # Eastleigh -> Eastleigh
    161808: 8662,  # Tonbridge Angels -> Tonbridge Angels
    161809: 7707,  # Carshalton Athletic -> Carshalton Athletic
    161812: 8663,  # Wealdstone -> Wealdstone
    161813: 4690,  # Dartford -> Dartford
    161816: 7656,  # Marine -> Marine
    161818: 7693,  # Ashton United -> Ashton United
    161822: 7754,  # FC United of Manchester -> United of Manchester
    161824: 7706,  # Buxton -> Buxton
    161831: 8660,  # Hemel Hempstead -> Hemel Hempstead Town
    161833: 8653,  # Gloucester City -> Gloucester City
    161834: 4710,  # Stourbridge -> Stourbridge
    161836: 4684,  # Oxford City -> Oxford City
    161838: 4682,  # Hitchin Town -> Hitchin Town
    161839: 7762,  # Yate Town -> Yate Town
    161841: 7696,  # Banbury United -> Banbury United
    162142: 2078,  # EIF -> EIF
    162146: 1163,  # Ilves -> Ilves
    162162: 689,  # SJK -> SJK
    162386: 949,  # Panetolikos -> Panetolikos
    163782: 7848,  # Mirassol -> Mirassol
    165545: 144,  # Atlético-GO -> Atletico Goianiense
    172341: 129,  # Ceará -> Ceara
    177063: 7207,  # Aveley -> Aveley
    177064: 7711,  # Cray Wanderers -> Cray Wanderers
    177067: 4699,  # Truro City -> Truro City
    178475: 173,  # RB Leipzig -> RB Leipzig
    181463: 11075,  # FC Rosengård -> Rosengård W
    181464: 16483,  # BK Häcken -> Häcken W
    181466: 11076,  # Djurgården -> Djurgården W
    181468: 15629,  # AIK -> AIK W
    181469: 15630,  # Hammarby IF -> Hammarby W
    181470: 11079,  # Kristianstads DFF -> Kristianstad W
    181471: 10900,  # Piteå IF -> Piteå W
    189475: 1578,  # Südtirol -> Sudtirol
    191433: 11659,  # IF Karlstad -> IF Karlstad
    191716: 1599,  # Philadelphia Union -> Philadelphia Union
    197815: 1193,  # Cuiabá -> Cuiaba
    207145: 4702,  # Chesham United -> Chesham United
    207242: 1612,  # Minnesota United -> Minnesota United FC
    208932: 527,  # Virtus Entella -> Virtus Entella
    212821: 4716,  # Casa Pia AC -> Casa Pia
    238635: 7738,  # Poole Town FC -> Poole Town
    241064: 12601,  # BK Olympic -> Olympic
    241285: 6703,  # Tvååkers IF -> Tvååker
    260273: 10669,  # PK-35 -> PK-35
    267810: 1598,  # Orlando City -> Orlando City SC
    274594: 7759,  # Wingate & Finchley -> Wingate & Finchley
    274599: 4679,  # Chorley -> Chorley
    274600: 7743,  # Rushall Olympic -> Rushall Olympic
    275027: 7612,  # AFC Totton -> AFC Totton
    275029: 7226,  # Frome Town -> Frome Town
    282326: 1844,  # Salford City -> Salford City
    282333: 4709,  # Stamford AFC -> Stamford
    282339: 7262,  # Whyteleafe -> Whyteleafe
    282340: 7217,  # Burgess Hill Town -> Burgess Hill Town
    282343: 7216,  # Brentwood Town -> Brentwood Town
    282351: 7761,  # Worthing -> Worthing
    282356: 7717,  # Gosport Borough -> Gosport Borough
    282358: 7758,  # Wimborne Town -> Wimborne Town
    282360: 7668,  # Sholing -> Sholing
    282365: 7714,  # Enfield Town -> Enfield Town
    282368: 8737,  # Chatham Town -> Chatham Town
    282369: 7708,  # Cheshunt -> Cheshunt
    282373: 4713,  # Whitehawk -> Whitehawk
    282375: 7735,  # Needham Market -> Needham Market
    282389: 7756,  # Warrington Town -> Warrington Town
    282390: 7740,  # Radcliffe -> Radcliffe
    282395: 7695,  # Bamber Bridge -> Bamber Bridge
    282396: 1839,  # AFC Fylde -> AFC Fylde
    282501: 7654,  # Maldon & Tiptree -> Maldon & Tiptree
    282502: 4711,  # Taunton Town -> Taunton Town
    282503: 7627,  # Chertsey Town -> Chertsey Town
    282505: 7257,  # Uxbridge -> Uxbridge
    282507: 8846,  # Leighton Town -> Leighton Town
    282508: 8149,  # Dulwich Hamlet -> Dulwich Hamlet
    282510: 7239,  # Leek Town -> Leek Town
    292923: 8822,  # Hebburn Town -> Hebburn Town
    292925: 8811,  # Hanworth Villa -> Hanworth Villa
    293352: 3012,  # Annecy FC -> Annecy
    300220: 11083,  # Vittsjö GIK -> Vittsjö W
    303470: 6663,  # Eskilsminne IF -> Eskilsminne
    303472: 6682,  # Karlbergs BK -> Karlberg
    307690: 1617,  # Portland Timbers -> Portland Timbers
    307691: 1603,  # Vancouver Whitecaps -> Vancouver Whitecaps
    338317: 7676,  # Three Bridges -> Three Bridges
    357259: 10139,  # AD Ceuta FC -> AD Ceuta FC
    418688: 6676,  # Hässleholms IF -> Hässleholms IF
    459581: 7750,  # Stratford Town -> Stratford Town
    488099: 5050,  # Kifisia FC -> Kifisia
    494050: 8157,  # FC Andorra -> FC Andorra
    514392: 11077,  # Eskilstuna United DFF -> Eskilstuna United W
    520517: 3851,  # Galway United FC -> Galway United
    546238: 1604,  # New York City FC -> New York City FC
    550910: 16008,  # IFK Norrköping -> Norrköping W
    554004: 11084,  # Växjö DFF -> Växjö W
    557101: 4707,  # Spennymoor Town FC -> Spennymoor Town
    575419: 6765,  # East Kilbride -> East Kilbride
    580382: 7745,  # Scarborough Athletic -> Scarborough Athletic
    580400: 7671,  # Spalding United -> Spalding United
    580402: 7631,  # Cleethorpes Town -> Cleethorpes Town
    580438: 8659,  # Dorking Wanderers -> Dorking Wanderers
    582838: 7645,  # Hanwell Town -> Hanwell Town
    583944: 870,  # Padova -> Padova
    584022: 509,  # Cesena -> Cesena
    626208: 8950,  # Stanway Rovers -> Stanway Rovers
    626250: 7619,  # Bracknell Town -> Bracknell Town
    627694: 12622,  # FBK Karlstad -> FBK Karlstad
    674289: 7737,  # Peterborough Sports -> Peterborough Sports
    674293: 8783,  # Farnham Town -> Farnham Town
    674304: 8899,  # Plymouth Parkway -> Plymouth Parkway
    675817: 6773,  # Kelty Hearts -> Kelty Hearts
    722265: 2242,  # FC Cincinnati -> FC Cincinnati
    741816: 9198,  # SJK Akatemia -> SJK Akatemia
    773958: 1608,  # Atlanta United -> Atlanta United FC
    799249: 4692,  # Hereford -> Hereford
    841094: 16597,  # FC Arlanda -> Arlanda
    865163: 7746,  # South Shields -> South Shields
    865168: 7692,  # Alvechurch -> Alvechurch
    867280: 1616,  # Los Angeles FC -> Los Angeles FC
    879829: 7214,  # Berkhamsted -> Berkhamsted
    885256: 2110,  # NFC Volos -> Volos NFC
    915807: 9569,  # Nashville SC -> Nashville SC
    916701: 6700,  # IFK Stocksund -> Stocksund
    953607: 7705,  # Bromsgrove Sporting -> Bromsgrove Sporting
    960127: 8904,  # Racing Club Warwick -> Racing Club Warwick
    960130: 8681,  # Anstey Nomads -> Anstey Nomads
    960720: 9568,  # Inter Miami CF -> Inter Miami
    1011931: 12585,  # FC Stockholm -> Stockholm Internazionale
    1022948: 2621,  # HFX Wanderers FC -> HFX Wanderers FC
    1022952: 2619,  # Cavalry FC -> Cavalry FC
    1022953: 2622,  # Pacific FC -> Pacific FC
    1022955: 3830,  # Forge FC -> Forge
    1071515: 7628,  # Chichester City FC -> Chichester City
    1074320: 15130,  # Estrela da Amadora -> Estrela
    1087122: 8953,  # Stockton Town -> Stockton Town
    1114695: 7744,  # Salisbury -> Salisbury
    1120766: 11082,  # IK Uppsala Fotboll -> Uppsala W
    1135780: 10169,  # Atlético Ottawa -> Atlético Ottawa
    1142849: 16004,  # Brommapojkarna -> Brommapojkarna W
    1144284: 16601,  # Nordic United FC -> United Nordic
    1186092: 8918,  # Warrington Rylands -> Rylands
    1218886: 16489,  # Austin FC -> Austin
    1218969: 5902,  # RAAL La Louviere -> RAAL La Louvière
    1221604: 531,  # Athletic Club -> Athletic Club
    1225382: 16598,  # FC Järfälla -> Järfälla
    1290988: 8690,  # Avro -> Avro
    1291005: 7236,  # Ilkeston Town -> Ilkeston Town
    1323940: 18310,  # Charlotte FC -> Charlotte
    1382661: 8812,  # Harborough Town FC -> Harborough Town
    1427963: 20787,  # St. Louis City -> St. Louis City
    1436202: 20881,  # Vancouver FC -> Vancouver FC
    1508925: 8870,  # Malvern Town -> Malvern Town
    1532123: 11936,  # Emley -> AFC Emley
    1598854: 21698,  # Malmö FF -> Malmö FF W
    1659622: 21712,  # Real Bedford -> Real Bedford
    1701119: 25484,  # San Diego FC -> San Diego
    1804164: 16994,  # Redcar Athletic -> Redcar Athletic
    1331261: 10598, # AFC Malmoe -> Ariana
    1786: 238, # Academico Viseu -> Academico Viseu
    9849: 876, # Arezzo -> Arezzo
    8355: 120, # Botafogo -> Botafogo
    161743: 9571, # Celta Fortuna -> Celta de Vigo II
    197693: 132, # Chapecoense AF -> Chapecoense-sc
    281467: 1009, # Erzurumspor -> Erzurumspor FK
    9824: 660, # FC Vaduz -> FC Vaduz
    1022954: 2623, # Inter Toronto FC -> York United
    8675: 17991, # Iraklis -> Iraklis 1908
    1585: 5046, # Kalamata -> Kalamata
    145007: 1584, # LR Vicenza -> Vicenza Vurtus
    1626: 1198, # Remo -> Remo
    4033: 9593, # Sabadell -> Sabadell
    8475: 738, # SK Beveren -> SK Beveren
    4249: 2147, # Strømmen -> Strommen
    1899648: 27461, # Supra du Quebec -> Supra du Quebec
    4496: 2116, # Thor Akureyri -> Thor Akureyri
    357274: 6343, # Çorum FK -> Çorum FK
}

SVENSKA_SPEL_TO_API_FOOTBALL_TEAMS = {
    1: 44,  # Burnley -> Burnley
    2: 50,  # Manchester City -> Manchester City
    5: 51,  # Brighton -> Brighton
    6: 71,  # Norwich -> Norwich
    11: 1109,  # Montenegro -> Montenegro
    12: 1117,  # Grekland -> Greece
    13: 1110,  # Andorra -> Andorra
    16: 1533,  # Kap Verde -> Cape Verde Islands
    17: 534,  # Las Palmas -> Las Palmas
    18: 537,  # Leganes -> Leganes
    21: 1096,  # Azerbajdzjan -> Azerbaijan
    22: 1092,  # Lettland -> Latvia
    23: 3,  # Kroatien -> Croatia
    24: 1103,  # Bulgarien -> Bulgaria
    31: 53,  # Reading -> Reading
    32: 75,  # Stoke -> Stoke City
    33: 58,  # Millwall -> Millwall
    34: 76,  # Swansea -> Swansea
    35: 1350,  # Peterborough -> Peterborough
    36: 1359,  # Luton -> Luton
    37: 59,  # Preston -> Preston
    38: 1356,  # Blackpool -> Blackpool
    39: 62,  # Sheffield U -> Sheffield Utd
    40: 72,  # Queens Park Rangers -> QPR
    41: 40,  # Liverpool -> Liverpool
    42: 211,  # Benfica -> Benfica
    43: 530,  # Atlético Madrid -> Atletico Madrid
    44: 68,  # Bolton -> Bolton
    45: 1355,  # Portsmouth -> Portsmouth
    48: 154,  # Fortaleza -> Fortaleza EC
    50: 529,  # Barcelona -> Barcelona
    51: 169,  # Frankfurt -> Eintracht Frankfurt
    52: 48,  # West Ham -> West Ham
    53: 80,  # Lyon -> Lyon
    54: 217,  # Braga -> SC Braga
    55: 257,  # Rangers -> Rangers
    56: 327,  # Bodö/Glimt -> Bodo/Glimt
    57: 497,  # Roma -> AS Roma
    58: 81,  # Marseille -> Marseille
    60: 46,  # Leicester -> Leicester
    62: 45,  # Everton -> Everton
    63: 60,  # West Bromwich -> West Brom
    64: 35,  # Bournemouth -> Bournemouth
    65: 533,  # Villarreal -> Villarreal
    66: 157,  # Bayern München -> Bayern München
    67: 70,  # Middlesbrough -> Middlesbrough
    68: 36,  # Fulham -> Fulham
    69: 65,  # Nottingham -> Nottingham Forest
    70: 1346,  # Coventry -> Coventry
    71: 49,  # Chelsea -> Chelsea
    72: 541,  # Real Madrid -> Real Madrid
    77: 173,  # RB Leipzig -> RB Leipzig
    78: 499,  # Atalanta -> Atalanta
    79: 38,  # Watford -> Watford
    80: 55,  # Brentford -> Brentford
    81: 41,  # Southampton -> Southampton
    82: 63,  # Leeds -> Leeds
    83: 42,  # Arsenal -> Arsenal
    84: 66,  # Aston Villa -> Aston Villa
    85: 33,  # Manchester United -> Manchester United
    86: 47,  # Tottenham -> Tottenham
    87: 52,  # Crystal Palace -> Crystal Palace
    88: 34,  # Newcastle -> Newcastle
    89: 39,  # Wolverhampton -> Wolves
    90: 43,  # Cardiff -> Cardiff
    91: 67,  # Blackburn -> Blackburn
    92: 64,  # Hull -> Hull City
    93: 54,  # Birmingham -> Birmingham
    94: 37,  # Huddersfield -> Huddersfield
    95: 747,  # Barnsley -> Barnsley
    96: 56,  # Bristol City -> Bristol City
    97: 69,  # Derby -> Derby
    98: 1362,  # Crawley Town -> Crawley Town
    99: 1819,  # Barrow -> Barrow
    100: 74,  # Sheffield W -> Sheffield Wednesday
    101: 1372,  # Cheltenham -> Cheltenham
    102: 73,  # Rotherham -> Rotherham
    103: 1335,  # Charlton -> Charlton
    104: 1379,  # Lincoln -> Lincoln
    105: 1336,  # Fleetwood -> Fleetwood Town
    106: 1338,  # Oxford -> Oxford United
    107: 1375,  # Morecambe -> Morecambe
    108: 1360,  # Accrington -> Accrington ST
    109: 748,  # Burton -> Burton Albion
    110: 57,  # Ipswich -> Ipswich
    111: 61,  # Wigan -> Wigan
    112: 1370,  # Cambridge -> Cambridge United
    113: 1337,  # Northampton -> Northampton
    114: 1343,  # Bradford -> Bradford
    115: 1339,  # Rochdale -> Rochdale
    116: 1342,  # Walsall -> Walsall
    117: 1844,  # Salford -> Salford City
    118: 1842,  # Harrogate -> Harrogate Town
    120: 1374,  # Mansfield -> Mansfield Town
    121: 1835,  # Sutton -> Sutton Utd
    122: 1373,  # Leyton Orient -> Leyton Orient
    123: 1364,  # Exeter -> Exeter City
    124: 1361,  # Colchester -> Colchester
    125: 1349,  # Oldham -> Oldham
    126: 1357,  # Plymouth -> Plymouth
    127: 1347,  # Gillingham -> Gillingham
    128: 1358,  # Wycombe -> Wycombe
    129: 746,  # Sunderland -> Sunderland
    130: 1352,  # Shrewsbury -> Shrewsbury
    131: 1354,  # Doncaster -> Doncaster
    132: 1371,  # Carlisle -> Carlisle
    133: 1368,  # Stevenage -> Stevenage
    134: 1353,  # Swindon -> Swindon Town
    135: 1367,  # Newport -> Newport County
    136: 1381,  # Tranmere -> Tranmere
    137: 1334,  # Bristol Rovers -> Bristol Rovers
    138: 1378,  # Forest Green -> Forest Green
    139: 1363,  # Crewe -> Crewe
    140: 1333,  # Wimbledon -> AFC Wimbledon
    141: 1351,  # Port Vale -> Port Vale
    142: 1348,  # Milton Keynes Dons -> Milton Keynes Dons
    143: 1366,  # Hartlepool -> Hartlepool
    145: 901,  # Partick Thistle -> Partick
    148: 172,  # Stuttgart -> VfB Stuttgart
    149: 165,  # Dortmund -> Borussia Dortmund
    159: 1063,  # Saint Etienne -> Saint Etienne
    160: 536,  # Sevilla -> Sevilla
    166: 415,  # Twente -> Twente
    167: 546,  # Getafe -> Getafe
    168: 3842,  # Sligo Rovers -> Sligo Rovers
    171: 228,  # Sporting Lissabon -> Sporting CP
    172: 490,  # Cagliari -> Cagliari
    173: 496,  # Juventus -> Juventus
    174: 9568,  # Inter Miami -> Inter Miami
    175: 1609,  # New England Revolution -> New England Revolution
    176: 1013,  # Grasshoppers -> Grasshoppers
    177: 644,  # Luzern -> FC Luzern
    179: 85,  # Paris Saint-Germain -> Paris Saint Germain
    180: 194,  # Ajax -> Ajax
    182: 124,  # Fluminense -> Fluminense
    183: 128,  # Santos -> Santos
    184: 1602,  # New York Red Bulls -> New York Red Bulls
    185: 1614,  # CF Montreal -> CF Montreal
    187: 1615,  # DC United -> DC United
    188: 539,  # Levante -> Levante
    190: 82,  # Montpellier -> Montpellier
    192: 201,  # AZ Alkmaar -> AZ Alkmaar
    193: 503,  # Torino -> Torino
    194: 489,  # Milan -> AC Milan
    195: 1062,  # Atletico Mineiro -> Atletico-MG
    196: 119,  # Internacional -> Internacional
    197: 120,  # Botafogo RJ -> Botafogo
    198: 131,  # Corinthians -> Corinthians
    199: 759,  # Viking -> Viking
    201: 2163,  # Värnamo -> IFK Varnamo
    202: 370,  # Sirius -> Sirius
    203: 500,  # Bologna -> Bologna
    205: 2241,  # Västerås -> Vasteras SK FK
    206: 376,  # Östersund -> Ostersunds FK
    207: 766,  # Halmstad -> Halmstad
    208: 2173,  # Norrby -> Norrby IF
    209: 375,  # Malmö -> Malmo FF
    210: 372,  # Elfsborg -> IF Elfsborg
    211: 728,  # Rayo Vallecano -> Rayo Vallecano
    212: 532,  # Valencia -> Valencia
    213: 395,  # Vejle -> Vejle
    215: 6702,  # Trollhättan -> Trollhättan
    216: 2170,  # GAIS -> Gais
    217: 740,  # Royal Antwerp -> Antwerp
    222: 250,  # Kilmarnock -> Kilmarnock
    223: 94,  # Rennes -> Rennes
    224: 91,  # Monaco -> Monaco
    229: 515,  # Spezia -> Spezia
    230: 505,  # Inter -> Inter
    232: 548,  # Real Sociedad -> Real Sociedad
    233: 543,  # Real Betis -> Real Betis
    234: 374,  # Kalmar -> Kalmar FF
    235: 2172,  # Degerfors -> Degerfors IF
    236: 367,  # Häcken -> BK Hacken
    237: 366,  # IFK Göteborg -> IFK Goteborg
    238: 811,  # Helsingborg -> Helsingborg
    240: 182,  # Union Berlin -> Union Berlin
    241: 531,  # Athletic Bilbao -> Athletic Club
    242: 538,  # Celta de Vigo -> Celta Vigo
    243: 540,  # Espanyol -> Espanyol
    246: 168,  # Bayer Leverkusen -> Bayer Leverkusen
    247: 84,  # Nice -> Nice
    248: 93,  # Reims -> Reims
    250: 95,  # Strasbourg -> Strasbourg
    252: 112,  # Metz -> Metz
    253: 83,  # Nantes -> Nantes
    255: 377,  # AIK -> AIK Stockholm
    256: 6692,  # Oskarshamn -> Oskarshamns AIK
    257: 6706,  # Utsikten -> Utsikten
    258: 765,  # AFC Eskilstuna -> AFC Eskilstuna
    259: 2176,  # Landskrona -> Landskrona BoIS
    262: 6699,  # Sollentuna -> Sollentuna
    263: 6705,  # Umeå -> Umeå FC
    266: 79,  # Lille -> Lille
    267: 116,  # Lens -> Lens
    268: 106,  # Brest -> Stade Brestois 29
    269: 363,  # Hammarby -> Hammarby FF
    270: 2240,  # Mjällby -> Mjallby AIF
    271: 365,  # Örebro -> Orebro SK
    272: 764,  # Jönköpings Södra -> Jonkopings Sodra
    275: 121,  # Palmeiras -> Palmeiras
    283: 127,  # Flamengo -> Flamengo
    288: 24,  # Polen -> Poland
    294: 134,  # Paranaense -> Atletico Paranaense
    299: 794,  # Bragantino -> RB Bragantino
    302: 247,  # Celtic -> Celtic
    305: 6701,  # Torn -> Torns
    306: 2165,  # Ljungskile -> Ljungskile SK
    307: 2162,  # Åtvidaberg -> Atvidabergs FF
    308: 6687,  # Lund -> Lund
    310: 6697,  # Sandviken -> Sandviken
    311: 799,  # Mirandes -> Mirandes
    314: 212,  # Porto -> FC Porto
    315: 216,  # Portimonense -> Portimonense
    316: 487,  # Lazio -> Lazio
    317: 731,  # Sporting Gijón -> Sporting Gijon
    318: 718,  # Real Oviedo -> Oviedo
    320: 175,  # Hamburg -> Hamburger SV
    322: 2184,  # Servette -> Servette FC
    324: 1603,  # Vancouver Whitecaps -> Vancouver Whitecaps
    327: 126,  # Sao Paulo -> Sao Paulo
    331: 549,  # Besiktas -> Beşiktaş
    332: 144,  # Goianiense -> Atletico Goianiense
    333: 368,  # Dalkurd -> Dalkurd FF
    334: 2175,  # Brage -> IK brage
    335: 2166,  # Örgryte -> Orgryte IS
    336: 2174,  # Öster -> Osters IF
    337: 400,  # FC Köpenhamn -> FC Copenhagen
    340: 554,  # Anderlecht -> Anderlecht
    342: 504,  # Verona -> Hellas Verona
    343: 492,  # Napoli -> Napoli
    345: 326,  # Vålerenga -> Valerenga
    347: 1579,  # Monza -> Monza
    349: 797,  # Elche -> Elche
    350: 160,  # Freiburg -> SC Freiburg
    351: 494,  # Udinese -> Udinese
    352: 514,  # Salernitana -> Salernitana
    353: 502,  # Fiorentina -> Fiorentina
    354: 2171,  # Varberg -> Varbergs BoIS FC
    355: 364,  # Djurgården -> Djurgardens IF
    356: 378,  # IFK Norrköping -> IFK Norrkoping
    357: 373,  # GIF Sundsvall -> GIF Sundsvall
    358: 727,  # Osasuna -> Osasuna
    366: 164,  # Mainz -> FSV Mainz 05
    373: 565,  # Young Boys -> BSC Young Boys
    374: 266,  # Mechelen -> KV Mechelen
    375: 736,  # Charleroi -> Charleroi
    380: 1600,  # Houston Dynamo -> Houston Dynamo
    381: 1601,  # Toronto FC -> Toronto FC
    382: 1598,  # Orlando City -> Orlando City SC
    383: 331,  # Rosenborg -> Rosenborg
    384: 329,  # Molde -> Molde
    387: 488,  # Sassuolo -> Sassuolo
    388: 2242,  # Cincinnati -> FC Cincinnati
    389: 6703,  # Tvååker -> Tvååker
    390: 6672,  # Haninge -> Haninge
    392: 813,  # Gefle -> Gefle IF
    393: 11673,  # Motala AIF -> Motala
    394: 252,  # Aberdeen -> Aberdeen
    395: 253,  # Dundee FC -> Dundee
    396: 1386,  # Dundee U -> Dundee Utd
    397: 256,  # Motherwell -> Motherwell
    401: 251,  # St. Mirren -> ST Mirren
    402: 254,  # Hearts -> Heart Of Midlothian
    403: 902,  # Ross County -> Ross County
    407: 321,  # Lilleström -> Lillestrom
    408: 133,  # Vasco da Gama -> Vasco DA Gama
    412: 397,  # Midtjylland -> FC Midtjylland
    414: 569,  # Club Brügge -> Club Brugge KV
    415: 401,  # Randers -> Randers FC
    416: 798,  # Mallorca -> Mallorca
    417: 645,  # Galatasaray -> Galatasaray
    419: 542,  # Alavés -> Alaves
    420: 333,  # Sarpsborg -> Sarpsborg 08 FF
    421: 330,  # Odds -> ODD Ballklubb
    424: 192,  # Köln -> 1. FC Köln
    426: 742,  # Genk -> Genk
    429: 9580,  # Burgos -> Burgos
    430: 723,  # Almería -> Almeria
    431: 1608,  # Atlanta United FC -> Atlanta United FC
    434: 1599,  # Philadelphia Union -> Philadelphia Union
    436: 147,  # Coritiba -> Coritiba
    442: 11663,  # Täby -> Täby
    443: 2168,  # Ängelholms FF -> Angelholms FF
    444: 12601,  # Olympic -> Olympic
    445: 12577,  # Vänersborgs FK -> Vänersborgs FK
    446: 11674,  # Oddevold -> Oddevold
    447: 11661,  # Sylvia -> Sylvia
    448: 11659,  # IF Karlstad -> IF Karlstad
    449: 16596,  # Hammarby TFF -> Hammarby Talang
    450: 812,  # Falkenberg -> Falkenbergs FF
    451: 6693,  # Qviding -> Qviding FIF
    454: 371,  # Brommapojkarna -> IF Brommapojkarna
    455: 11666,  # BK Forward -> Forward
    456: 11671,  # Piteå -> Piteå
    457: 12585,  # Stockholm Internazionale -> Stockholm Internazionale
    458: 398,  # Nordsjälland -> FC Nordsjaelland
    459: 6708,  # Vasalund -> Vasalund
    461: 402,  # Aalborg -> Aalborg
    468: 96,  # Toulouse -> Toulouse
    473: 111,  # Le Havre -> Le Havre
    489: 328,  # Haugesund -> Haugesund
    490: 320,  # Kristiansund -> Kristiansund BK
    491: 324,  # Strömsgodset -> Stromsgodset
    492: 332,  # Sandefjord -> Sandefjord
    493: 2159,  # Ham-Kam -> Ham-Kam
    496: 895,  # Como -> Como
    497: 520,  # Cremonese -> Cremonese
    498: 1393,  # Royale Union SG -> Union St. Gilloise
    499: 174,  # Schalke 04 -> FC Schalke 04
    501: 1011,  # St. Gallen -> FC ST. Gallen
    504: 6698,  # Skövde AIK -> Skövde AIK
    512: 6676,  # Hässleholms IF -> Hässleholms IF
    515: 12574,  # Nordvärmland -> Nordvärmland
    522: 545,  # Eibar -> Eibar
    530: 152,  # Juventude -> Juventude
    531: 1193,  # Cuiaba Esporte -> Cuiaba
    532: 275,  # Stjarnan -> Stjarnan
    534: 1595,  # Seattle Sounders -> Seattle Sounders
    535: 1612,  # Minnesota United -> Minnesota United FC
    537: 11675,  # Kristianstad -> Kristianstad
    538: 732,  # Real Zaragoza -> Zaragoza
    540: 271,  # KR Reykjavik -> KR Reykjavik
    543: 2117,  # Fram -> Fram Reykjavik
    544: 278,  # Vikingur -> Vikingur Reykjavik
    545: 276,  # Breidablik -> Breidablik
    546: 369,  # Trelleborg -> Trelleborg
    550: 11675,  # Kristianstad -> Kristianstad
    551: 1014,  # Lausanne -> Lausanne
    552: 630,  # Sion -> FC Sion
    553: 606,  # Lugano -> FC Lugano
    554: 16489,  # Austin FC -> Austin
    561: 6671,  # Gute -> Gute
    564: 12617,  # Räppe -> Räppe
    565: 11664,  # Örebro Syrianska -> Örebro Syrianska
    567: 12586,  # Friska Viljor -> Friska Viljor
    568: 6675,  # Husqvarna -> Husqvarna
    570: 16600,  # Mjölby -> Mjölby
    573: 6700,  # Stocksund -> Stocksund
    574: 547,  # Girona -> Girona
    575: 9390,  # Ibiza -> Ibiza
    576: 1613,  # Columbus Crew -> Columbus Crew
    577: 551,  # Basel -> FC Basel 1893
    579: 6663,  # Eskilsminne -> Eskilsminne
    582: 135,  # Cruzeiro -> Cruzeiro
    588: 6694,  # Rosengård -> Rosengård
    590: 325,  # Tromsö -> Tromso
    591: 801,  # Pisa -> Pisa
    592: 529,  # Barcelona -> Barcelona
    593: 80,  # Lyon -> Lyon
    599: 12570,  # Ahlafors -> Ahlafors
    604: 16,  # Mexiko -> Mexico
    607: 12614,  # IFK Hässleholm -> IFK Hässleholm
    610: 12620,  # Torslanda -> Torslanda
    612: 12609,  # Västra Frölunda -> Västra Frölunda
    613: 6667,  # Gauthiod -> Gauthiod
    615: 12612,  # Berga -> Berga
    619: 12606,  # Onsala -> Onsala
    630: 745,  # Kaiserslautern -> 1. FC Kaiserslautern
    636: 1345,  # Chesterfield -> Chesterfield
    639: 2077,  # Oulu -> AC Oulu
    640: 689,  # SJK -> SJK
    641: 2080,  # Haka -> Haka
    642: 1166,  # Lahti -> Lahti
    643: 1164,  # Inter Åbo -> Inter Turku
    644: 1163,  # Ilves -> Ilves
    648: 653,  # Cork -> Cork City
    661: 270,  # Hafnarfjördur -> FH hafnarfjordur
    663: 130,  # Gremio -> Gremio
    668: 7,  # Uruguay -> Uruguay
    669: 2382,  # Ecuador -> Ecuador
    670: 1104,  # Georgien -> Georgia
    672: 770,  # Tjeckien -> Czechia
    673: 15,  # Schweiz -> Switzerland
    674: 1090,  # Norge -> Norway
    675: 14,  # Serbien -> Serbia
    676: 771,  # Nordirland -> Northern Ireland
    677: 1,  # Belgien -> Belgium
    678: 1118,  # Nederländerna -> Netherlands
    679: 776,  # Irland -> Rep. Of Ireland
    680: 1094,  # Armenien -> Armenia
    681: 1106,  # Cypern -> Cyprus
    682: 1111,  # Kosovo -> Kosovo
    683: 9,  # Spanien -> Spain
    684: 27,  # Portugal -> Portugal
    685: 778,  # Albanien -> Albania
    687: 18,  # Island -> Iceland
    688: 1116,  # Israel -> Israel
    689: 5,  # Sverige -> Sweden
    690: 1091,  # Slovenien -> Slovenia
    692: 775,  # Österrike -> Austria
    693: 2,  # Frankrike -> France
    694: 21,  # Danmark -> Denmark
    696: 1114,  # Moldavien -> Moldova
    697: 10,  # England -> England
    698: 769,  # Ungern -> Hungary
    699: 1097,  # Litauen -> Lithuania
    700: 1102,  # Luxemburg -> Luxembourg
    701: 1101,  # Estland -> Estonia
    702: 1105,  # Nordmakedonien -> FYR Macedonia
    703: 1099,  # Finland -> Finland
    704: 1113,  # Bosnien & Hercegovina -> Bosnia & Herzegovina
    705: 30,  # Peru -> Peru
    706: 4673,  # Nya Zeeland -> New Zealand
    707: 8,  # Colombia -> Colombia
    708: 23,  # Saudiarabien -> Saudi Arabia
    709: 2384,  # USA -> USA
    710: 5529,  # Kanada -> Canada
    711: 22,  # Iran -> Iran
    712: 6,  # Brasilien -> Brazil
    713: 12,  # Japan -> Japan
    714: 2383,  # Chile -> Chile
    716: 768,  # Italien -> Italy
    717: 25,  # Tyskland -> Germany
    718: 774,  # Rumänien -> Romania
    719: 777,  # Turkiet -> Türkiye
    720: 1098,  # Färöarna -> Faroe Islands
    721: 1112,  # Malta -> Malta
    722: 24,  # Polen -> Poland
    723: 773,  # Slovakien -> Slovakia
    724: 767,  # Wales -> Wales
    725: 1108,  # Skottland -> Scotland
    726: 772,  # Ukraina -> Ukraine
    734: 20,  # Australien -> Australia
    739: 31,  # Marocko -> Morocco
    741: 32,  # Egypten -> Egypt
    747: 28,  # Tunisien -> Tunisia
    755: 26,  # Argentina -> Argentina
    758: 6682,  # Karlberg -> Karlberg
    761: 1501,  # Elfenbenskusten -> Ivory Coast
    774: 1532,  # Algeriet -> Algeria
    785: 1365,  # Grimsby -> Grimsby
    801: 27,  # Portugal -> Portugal
    807: 29,  # Costa Rica -> Costa Rica
    813: 272,  # Akureyrar -> KA Akureyri
    817: 649,  # HJK Helsingfors -> HJK Helsinki
    819: 650,  # VPS -> VPS
    821: 16598,  # Järfälla -> Järfälla
    823: 268,  # Vestmannaeyja -> IBV Vestmannaeyjar
    828: 375,  # AFC Malmö -> Malmo FF
    829: 12575,  # Stenungsund -> Stenungsund
    830: 1165,  # KuPS -> KuPS
    831: 587,  # Mariehamn -> Mariehamn
    832: 1169,  # Honka -> Honka
    841: 827,  # Akranes -> IA Akranes
    847: 2142,  # Bryne -> Bryne
    850: 319,  # Brann -> Brann
    851: 322,  # Ranheim -> Ranheim
    852: 2143,  # KFUM Oslo -> KFUM Oslo
    853: 758,  # Sogndal -> Sogndal
    858: 21,  # Danmark -> Denmark
    862: 10,  # England -> England
    863: 1118,  # Nederländerna -> Netherlands
    864: 118,  # Bahia -> Bahia
    866: 27,  # Portugal -> Portugal
    875: 323,  # Stabæk -> Stabaek
    897: 5,  # Sverige -> Sweden
    905: 767,  # Wales -> Wales
    907: 25,  # Tyskland -> Germany
    909: 2,  # Frankrike -> France
    917: 2153,  # Raufoss -> Raufoss
    1004: 260,  # OH Leuven -> OH Leuven
    1008: 2149,  # Fredrikstad FK -> Fredrikstad
    1009: 334,  # IK Start -> Start
    1010: 261,  # Westerlo -> KVC Westerlo
    1019: 733,  # Standard Liege -> Standard Liege
    1022: 735,  # St. Truidense -> St. Truiden
    1024: 625,  # Lyngby -> Lyngby
    1029: 1123,  # Aris -> Aris Thessalonikis
    1039: 4686,  # Stockport -> Stockport County
    1047: 617,  # Panathinaikos -> Panathinaikos
    1056: 523,  # Parma -> Parma
    1061: 867,  # Lecce -> Lecce
    1062: 162,  # Werder Bremen -> Werder Bremen
    1066: 9595,  # Villarreal B -> Villarreal II
    1214: 722,  # Albacete -> Albacete
    1294: 773,  # Slovakien -> Slovakia
    1298: 770,  # Tjeckien -> Czechia
    1301: 16601,  # Nordic United -> United Nordic
    1412: 1822,  # Eastleigh -> Eastleigh
    1480: 1569,  # Qatar -> Qatar
    1488: 6763,  # Cove Rangers -> Cove Rangers
    1547: 1369,  # Barnet -> Barnet
    1554: 1837,  # Wrexham -> Wrexham
    1557: 161,  # Wolfsburg -> VfL Wolfsburg
    1606: 1389,  # Falkirk -> Falkirk
    2005: 1376,  # Notts County -> Notts County
    3408: 5530,  # Curacao -> Curaçao
    3471: 16597,  # Arlanda -> Arlanda
    3490: 21309,  # Zenith -> Zenith
    3491: 20835,  # Farsta -> Farsta
    3534: 2113,  # Kopavogur -> HK Kopavogur
    3808: 19000,  # Simrishamn -> Simrishamn
    3809: 12657,  # Laholm -> Laholm
    3963: 20838,  # Viggbyholm -> Viggbyholms IK
    3964: 2150,  # Hödd -> hodd
    3965: 2144,  # Kongsvinger -> Kongsvinger
    3966: 7012,  # Moss -> Moss
    4148: 2076,  # Kotka -> Kooteepee
    4193: 11,  # Panama -> Panama
    4303: 16605,  # IFK Skövde -> IFK Skövde
    4357: 25,  # Tyskland -> Germany
    4358: 15,  # Schweiz -> Switzerland
    4359: 768,  # Italien -> Italy
    4360: 2,  # Frankrike -> France
    4361: 9,  # Spanien -> Spain
    4362: 774,  # Rumänien -> Romania
    4363: 1104,  # Georgien -> Georgia
    4397: 2386,  # Haiti -> Haiti
    4443: 7848,  # Mirassol -> Mirassol
    5042: 1687,  # Catanzaro -> Catanzaro
    5051: 24,  # Polen -> Poland
    5366: 21595,  # Avs Futebol Sad -> AVS
    6808: 2078,  # Ekenas Idrottsforening -> EIF
    6878: 2082,  # IF Gnistan -> Gnistan
    6982: 4165,  # IF Vestri -> Vestri
    7149: 22935,  # IK Kongahälla -> Kongahälla
    7429: 263,  # Beerschot -> Beerschot VA
    7775: 5254,  # Castellón -> Castellón
    9037: 14543,  # Åstorps FF -> Åstorp
    9202: 2075,  # FF Jaro -> FF Jaro
    9635: 5902,  # Raal La Louviere -> RAAL La Louvière
    11255: 1168,  # TPS -> Turku PS
    298: 2384,  # America -> USA
    1540: 2381, # Bolivia -> Bolivia
    737: 2379, # Venezuela -> Venezuela
    309: 11662, # Team TG -> Umea FF
    312: 136, # Vitória de Guimarães -> Vitória
}
