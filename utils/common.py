import re
from enum import Enum
from typing import Literal

LEAGUES_EXTERNAL_IDS = ["39","40","41","42","43","179","180","183","184","78","79","135","136","140","141","61","62","88","144","94","203","197","113","114","563","564","593","597","595","592","594","549","736","45"]



FOTMOB_TO_API_FOOTBALL_TEAM_MAPPING = {
    8678: 35,  # AFC Bournemouth
    9825: 42,  # Arsenal
    10252: 66,  # Aston Villa
    9937: 55,  # Brentford
    10204: 51,  # Brighton & Hove Albion
    8455: 49,  # Chelsea
    8669: 1346,  # Coventry City
    9826: 52,  # Crystal Palace
    8668: 45,  # Everton
    9879: 36,  # Fulham
    8667: 64,  # Hull City
    9902: 57,  # Ipswich Town
    8463: 63,  # Leeds United
    8650: 40,  # Liverpool
    8456: 50,  # Manchester City
    10260: 33,  # Manchester United
    10261: 34,  # Newcastle United
    10203: 65,  # Nottingham Forest
    8472: 746,  # Sunderland
    8586: 47,  # Tottenham Hotspur
    10172: 72,  # Queens Park Rangers
    10004: 58,  # Millwall
    10003: 76,  # Swansea City
    8659: 60,  # West Bromwich Albion
    8559: 68,  # Bolton Wanderers
    8451: 1335,  # Charlton Athletic
    8549: 70,  # Middlesbrough
    9817: 38,  # Watford
    8655: 67,  # Blackburn Rovers
    8654: 48,  # West Ham United
    8191: 44,  # Burnley
    8602: 39,  # Wolverhampton Wanderers
    9841: 1837,  # Wrexham
    8344: 43,  # Cardiff City
    8658: 54,  # Birmingham City
    8657: 62,  # Sheffield United
    10170: 69,  # Derby County
    8430: 1379,  # Lincoln City
    8411: 59,  # Preston North End
    9850: 71,  # Norwich City
    10194: 75,  # Stoke City
    8462: 1355,  # Portsmouth
    8427: 56,  # Bristol City
    8466: 41,  # Southampton
    8484: 1343,  # Bradford City
    9796: 37,  # Huddersfield Town
    10007: 4686,  # Stockport County
    8346: 1359,  # Luton Town
    9834: 1370,  # Cambridge United
    9818: 1374,  # Mansfield Town
    45729: 1832,  # Bromley
    10163: 74,  # Sheffield Wednesday
    8645: 1348,  # Milton Keynes Dons
    8653: 1338,  # Oxford United
    8197: 46,  # Leicester City
    10253: 1368,  # Stevenage
    8676: 1358,  # Wycombe Wanderers
    8483: 1356,  # Blackpool
    9792: 748,  # Burton Albion
    9819: 1376,  # Notts County
    9798: 53,  # Reading
    8528: 61,  # Wigan Athletic
    9903: 1354,  # Doncaster Rovers
    8351: 1373,  # Leyton Orient
    8283: 747,  # Barnsley
    8401: 1357,  # Plymouth Argyle
    8677: 1350,  # Peterborough United
    158319: 1333,  # AFC Wimbledon
    10006: 1342,  # Walsall
    10262: 1367,  # Newport County
    8175: 1369,  # Barnet
    9785: 1349,  # Oldham Athletic
    8313: 1381,  # Tranmere Rovers
    9916: 1828,  # York City
    8680: 1372,  # Cheltenham Town
    9784: 1363,  # Crewe Alexandra
    45723: 1336,  # Fleetwood Town
    10005: 1365,  # Grimsby Town
    8416: 1361,  # Colchester United
    8671: 1360,  # Accrington Stanley
    8651: 1337,  # Northampton Town
    9795: 1353,  # Swindon Town
    10104: 1334,  # Bristol Rovers
    8119: 73,  # Rotherham United
    9786: 1345,  # Chesterfield
    8647: 1362,  # Crawley Town
    9833: 1364,  # Exeter City
    282326: 1844,  # Salford City
    9799: 1351,  # Port Vale
    9896: 1352,  # Shrewsbury Town
    10173: 1347,  # Gillingham
    8493: 1339,  # Rochdale
    7946: 1842,  # Harrogate Town
    4047: 7690,  # Hornchurch
    158316: 1835,  # Sutton United
    8652: 1341,  # Southend United
    8488: 1366,  # Hartlepool United
    9828: 1378,  # Forest Green Rovers
    2488: 1830,  # Boreham Wood
    282396: 1839,  # AFC Fylde
    6189: 1823,  # Gateshead FC
    8646: 8146,  # Boston United
    282351: 7761,  # Worthing
    161803: 1822,  # Eastleigh
    8465: 1818,  # Aldershot Town
    10284: 4695,  # Kidderminster Harriers
    10198: 1377,  # Yeovil Town
    10254: 7752,  # Tamworth
    10196: 1371,  # Carlisle United
    8412: 1340,  # Scunthorpe United
    161812: 8663,  # Wealdstone
    10195: 1841,  # FC Halifax Town
    9915: 4700,  # Altrincham
    6350: 1819,  # Barrow
    161801: 1834,  # Solihull Moors
    8345: 1836,  # Woking

    8121: 77,  # Angers
    8583: 108,  # Auxerre
    8521: 106,  # Brest
    9746: 111,  # Le Havre
    8682: 1298,  # Le Mans
    8588: 116,  # Lens
    8639: 79,  # Lille
    8689: 97,  # Lorient
    9748: 80,  # Lyon
    8592: 81,  # Marseille
    9829: 91,  # Monaco
    9831: 84,  # Nice
    6379: 114,  # Paris FC
    9847: 85,  # Paris Saint-Germain
    9851: 94,  # Rennes
    9848: 95,  # Strasbourg
    9941: 96,  # Toulouse
    10242: 110,  # Troyes
    9853: 1063,  # Saint-Etienne
    293352: 3012,  # Annecy FC
    47214: 1304,  # Dunkerque
    10249: 82,  # Montpellier
    8550: 112,  # Metz
    6390: 104,  # Red Star
    4120: 1301,  # Rodez
    9747: 90,  # Guingamp
    6355: 1297,  # Pau
    7853: 433,  # Laval
    9837: 93,  # Reims
    8481: 102,  # Nancy
    4170: 1299,  # Boulogne
    9836: 89,  # Dijon
    9855: 101,  # Grenoble
    8311: 99,  # Clermont Foot
    9874: 115,  # Sochaux
    9830: 83,  # Nantes

    8406: 170,  # Augsburg
    8178: 168,  # Bayer Leverkusen
    9823: 157,  # Bayern München
    9789: 165,  # Borussia Dortmund
    9788: 163,  # Borussia Mönchengladbach
    9810: 169,  # Eintracht Frankfurt
    8232: 1660,  # Elversberg
    8722: 192,  # 1. FC Köln
    8358: 160,  # Freiburg
    9790: 175,  # Hamburger SV
    8226: 167,  # Hoffenheim
    9905: 164,  # Mainz 05
    8460: 185,  # Paderborn
    178475: 173,  # RB Leipzig
    10189: 174,  # Schalke 04
    8149: 182,  # Union Berlin
    10269: 172,  # VfB Stuttgart
    8697: 162,  # Werder Bremen
    8165: 171,  # 1. FC Nürnberg
    8177: 159,  # Hertha BSC
    8295: 785,  # Karlsruher SC
    8721: 161,  # Wolfsburg
    9776: 744,  # Eintracht Braunschweig
    9912: 188,  # Arminia Bielefeld
    9911: 176,  # Bochum
    94937: 180,  # FC Heidenheim
    8188: 179,  # Magdeburg
    8480: 183,  # Dynamo Dresden
    8150: 191,  # Holstein Kiel
    8152: 186,  # St. Pauli
    8350: 745,  # Kaiserslautern
    8262: 181,  # Darmstadt
    8357: 178,  # Greuther Fürth
    9904: 166,  # Hannover 96
    9775: 1324,  # VfL Osnabrück

    10229: 201,  # AZ Alkmaar
    8674: 202,  # FC Groningen
    6433: 410,  # Go Ahead Eagles
    6422: 205,  # Fortuna Sittard
    8593: 194,  # Ajax
    8640: 197,  # PSV Eindhoven
    10235: 209,  # Feyenoord
    10228: 210,  # SC Heerenveen
    10218: 196,  # Excelsior
    8464: 413,  # NEC Nijmegen
    8611: 415,  # FC Twente
    8614: 426,  # Sparta Rotterdam
    6414: 427,  # Telstar
    9908: 207,  # FC Utrecht
    6413: 193,  # PEC Zwolle
    10217: 198,  # ADO Den Haag
    8525: 195,  # Willem II
    7788: 420,  # Cambuur

    158085: 240,  # Arouca
    9773: 212,  # FC Porto
    10212: 214,  # Maritimo
    9772: 211,  # Benfica
    10214: 225,  # Nacional
    9768: 228,  # Sporting CP
    1567: 227,  # Santa Clara
    9764: 762,  # Gil Vicente
    1074320: 15130,  # Estrela da Amadora
    10264: 217,  # Braga
    1634: 242,  # Famalicao
    9780: 4724,  # Alverca
    7842: 230,  # Estoril
    8348: 215,  # Moreirense
    7844: 224,  # Vitoria de Guimaraes
    7841: 226,  # Rio Ave
    212821: 4716,  # Casa Pia AC

    6694: 370,  # Sirius
    8248: 363,  # Hammarby
    8428: 367,  # Häcken
    8349: 377,  # AIK
    8014: 372,  # Elfsborg
    9802: 364,  # Djurgården
    10237: 375,  # Malmö FF
    6194: 2241,  # Västerås SK
    8297: 2170,  # GAIS
    8501: 371,  # Brommapojkarna
    8127: 2240,  # Mjällby
    9893: 366,  # IFK Göteborg
    9892: 374,  # Kalmar FF
    6544: 2172,  # Degerfors
    10002: 2166,  # Örgryte
    8310: 766,  # Halmstads BK
    8449: 378,  # IFK Norrköping
    6545: 812,  # Falkenbergs FF
    6692: 2171,  # Varbergs BoIS FC
    2004: 376,  # Östersunds FK
    8511: 2176,  # Landskrona BoIS
    1144284: 16601,  # Nordic United FC
    8641: 2174,  # Östers IF
    6241: 6697,  # Sandvikens IF
    6160: 11674,  # IK Oddevold
    9859: 811,  # Helsingborg
    2014: 2173,  # Norrby
    6690: 2175,  # IK Brage
    8500: 2165,  # Ljungskile
    6181: 2163,  # IFK Värnamo
    8527: 365,  # Örebro
    8359: 373,  # GIF Sundsvall

    8564: 489,  # Milan
    8524: 499,  # Atalanta
    9857: 500,  # Bologna
    8529: 490,  # Cagliari
    10171: 895,  # Como
    8535: 502,  # Fiorentina
    9891: 512,  # Frosinone
    10233: 495,  # Genoa
    8636: 505,  # Inter
    9885: 496,  # Juventus
    8543: 487,  # Lazio
    9888: 867,  # Lecce
    6504: 1579,  # Monza
    10167: 523,  # Parma
    8686: 497,  # Roma
    9875: 492,  # Napoli
    7943: 488,  # Sassuolo
    9804: 503,  # Torino
    8600: 494,  # Udinese
    7881: 517,  # Venezia
    8522: 507,  # Ascoli
    6722: 528,  # Avellino
    6266: 506,  # Benevento
    583944: 870,  # Padova
    6488: 1581,  # Carrarese
    10168: 1687,  # Catanzaro
    584022: 509,  # Cesena
    7801: 520,  # Cremonese
    8534: 511,  # Empoli
    9876: 504,  # Hellas Verona
    6106: 863,  # Juve Stabia
    9889: 1693,  # Mantova
    9887: 899,  # Modena
    8540: 522,  # Palermo
    6479: 801,  # Pisa
    9882: 498,  # Sampdoria
    189475: 1578,  # Südtirol
    208932: 527,  # Virtus Entella

    9866: 542,  # Deportivo Alaves
    8558: 540,  # Espanyol
    9906: 530,  # Atletico Madrid
    8302: 536,  # Sevilla
    8696: 4665,  # Racing Santander
    10205: 533,  # Villarreal
    9783: 544,  # Deportivo A Coruña
    10268: 797,  # Elche
    8370: 728,  # Rayo Vallecano
    8315: 531,  # Athletic Club
    8634: 529,  # Barcelona
    9910: 538,  # Celta Vigo
    8371: 727,  # Osasuna
    8603: 543,  # Real Betis
    8633: 541,  # Real Madrid
    8560: 548,  # Real Sociedad
    10267: 532,  # Valencia
    9864: 535,  # Malaga
    8305: 546,  # Getafe
    8581: 539,  # Levante
    494050: 8157,  # FC Andorra
    9865: 723,  # Almeria
    9867: 719,  # Tenerife
    8661: 798,  # Mallorca
    7876: 9580,  # Burgos CF
    8306: 534,  # Las Palmas
    10279: 5254,  # Castellon
    7732: 547,  # Girona
    7854: 537,  # Leganes
    8385: 724,  # Cadiz
    7878: 715,  # Granada
    8670: 718,  # Real Oviedo
    9869: 731,  # Sporting Gijon
    7869: 713,  # Cordoba
    8393: 722,  # Albacete
    161744: 9585,  # Real Sociedad B
    8372: 545,  # Eibar
    10281: 720,  # Real Valladolid
    8288: 9692,  # Eldense
    357259: 10139,  # AD Ceuta FC

    8342: 569,  # Club Brugge
    9988: 740,  # Royal Antwerp
    9991: 631,  # Gent
    9986: 736,  # Sporting Charleroi
    7978: 1393,  # Union St.Gilloise
    10000: 600,  # Zulte Waregem
    9987: 742,  # Genk
    8475: 8475,  # SK Beveren
    8635: 554,  # Anderlecht
    9985: 733,  # Standard Liege
    9984: 741,  # Cercle Brugge
    9997: 735,  # St.Truiden
    6702: 259,  # Lommel
    8203: 266,  # KV Mechelen
    1218969: 5902,  # RAAL La Louviere
    10001: 261,  # Westerlo
    1773: 260,  # OH Leuven
    8571: 734,  # Kortrijk

    9925: 247,  # Celtic
    9800: 251,  # St. Mirren
    9927: 256,  # Motherwell
    9860: 254,  # Heart of Midlothian
    8284: 253,  # Dundee FC
    8467: 258,  # St. Johnstone
    10251: 249,  # Hibernian
    8485: 252,  # Aberdeen
    8548: 257,  # Rangers
    8596: 1389,  # Falkirk
    9938: 1386,  # Dundee United
    8597: 250,  # Kilmarnock
    8426: 901,  # Partick Thistle
    9913: 1387,  # Ayr United
    8143: 6784,  # Stenhousemuir
    10250: 1385,  # Raith Rovers
    8282: 255,  # Livingston
    8457: 1388,  # Dunfermline Athletic
    8066: 903,  # Inverness Caledonian Thistle
    8280: 4249,  # Arbroath
    8235: 6778,  # Queen's Park
    8648: 1383,  # Greenock Morton
    8429: 248,  # Hamilton Academical
    575419: 6765,  # East Kilbride
    8649: 902,  # Ross County
    8158: 6777,  # Peterhead
    8160: 1391,  # Alloa Athletic
    9758: 6764,  # East Fife
    8176: 4668,  # Airdrieonians
    6002: 6763,  # Cove Rangers
    9924: 1384,  # Queen of the South
    8257: 4251,  # Montrose
    9914: 6767,  # Elgin City
    8145: 4250,  # Forfar Athletic
    8409: 6762,  # Clyde
    6661: 6781,  # Spartans FC
    46036: 6757,  # Annan Athletic
    8141: 4669,  # Stirling Albion
    9926: 6785,  # Stranraer
    675817: 6773,  # Kelty Hearts
    8321: 1382,  # Dumbarton
    4130: 6766,  # Edinburgh City

    8563: 575,  # AEK Athens
    8239: 1123,  # Aris Thessaloniki
    80654: 955,  # Asteras Tripolis
    10187: 12260,  # Atromitos
    8675: 8675,  # Iraklis
    1585: 1585,  # Kalamata
    488099: 5050,  # Kifisia FC
    4493: 957,  # Levadiakos
    885256: 2110,  # NFC Volos
    7753: 7753,  # OFI Crete
    8638: 8638,  # Olympiacos
    8619: 8619,  # PAOK Thessaloniki
    10200: 617,  # Panathinaikos
    162386: 949,  # Panetolikos

    1933: 564,  # Başakşehir
    7800: 997,  # Gençlerbirliği
    10188: 549,  # Beşiktaş
    2166: 1007,  # Rizespor
    1925: 994,  # Göztepe
    9750: 3603,  # Samsunspor
    8637: 645,  # Galatasaray
    4678: 996,  # Alanyaspor
    4081: 3573,  # Gaziantep FK
    4685: 1004,  # Kasımpaşa
    9752: 998,  # Trabzonspor
    8695: 611,  # Fenerbahçe
    4681: 3588,  # Eyüpspor
    8622: 607,  # Konyaspor
    1569: 7411,  # Kocaelispor

    181464: 16483,  # BK Häcken
    181469: 15630,  # Hammarby IF
    1598854: 21698,  # Malmö FF
    181468: 15629,  # AIK
    514392: 11077,  # Eskilstuna United DFF
    181470: 11079,  # Kristianstads DFF
    300220: 11083,  # Vittsjö GIK
    181471: 10900,  # Piteå IF
    1142849: 16004,  # Brommapojkarna
    554004: 11084,  # Växjö DFF
    181466: 11076,  # Djurgården
    550910: 16008,  # IFK Norrköping
    181463: 11075,  # FC Rosengård
    1120766: 11082,  # IK Uppsala Fotboll

    4438: 16596,  # Hammarby TFF
    191433: 11659,  # IF Karlstad
    1011931: 12585,  # FC Stockholm
    841094: 16597,  # FC Arlanda
    6238: 765,  # AFC Eskilstuna
    627694: 12622,  # FBK Karlstad
    8425: 12578,  # Enköping
    303472: 6682,  # Karlbergs BK
    10225: 2169,  # Assyriska FF
    1967: 6708,  # Vasalunds IF
    7997: 813,  # Gefle
    111120: 6699,  # Sollentuna FK
    8601: 6705,  # Umeå
    1225382: 16598,  # FC Järfälla
    916701: 6700,  # IFK Stocksund
    2006: 11671,  # Piteå
    8333: 369,  # Trelleborgs FF
    6622: 2162,  # Åtvidaberg
    418688: 6676,  # Hässleholms IF
    6174: 6687,  # Lunds BK
    6170: 6694,  # FC Rosengård
    241285: 6703,  # Tvååkers IF
    303470: 6663,  # Eskilsminne IF
    241064: 12601,  # BK Olympic
    8510: 764,  # Jönköping S.
    6183: 6702,  # Trollhättan FC
    8189: 2168,  # Ängelholms FF
    1965: 11675,  # Kristianstad FC
    6153: 6698,  # Skövde AIK
    73158: 6706,  # Utsiktens BK
}

FOTMOB_TO_API_FOOTBALL_LEAGUE_MAPPING = {
    39: 47,       # Premier League
    40: 48,       # Championship
    41: 108,      # League One
    42: 109,      # League Two
    43: 117,      # National League
    45: 132,      # FA Cup
    61: 53,       # Ligue 1
    62: 110,      # Ligue 2
    78: 54,       # Bundesliga -> 1. Bundesliga
    79: 146,      # 2. Bundesliga
    88: 57,       # Eredivisie
    94: 61,       # Primeira Liga -> Liga Portugal
    113: 67,      # Allsvenskan
    114: 168,     # Superettan
    135: 55,      # Serie A
    136: 86,      # Serie B
    140: 87,      # La Liga -> LaLiga
    141: 140,     # Segunda División -> LaLiga2
    144: 40,      # Jupiler Pro League -> First Division A
    179: 64,      # Premiership
    180: 123,     # Championship
    183: 124,     # League One
    184: 125,     # League Two
    197: 135,     # Super League 1
    203: 71,      # Süper Lig -> Super Lig
    549: 9089,    # Damallsvenskan -> Damallsvenskan (W)
    563: 169,     # Ettan - Norra -> Ettan
    564: 169,     # Ettan - Södra -> Ettan

    592: None,    # Division 2 - Norra Götaland
    593: None,    # Division 2 - Norra Svealand
    594: None,    # Division 2 - Norrland
    595: None,    # Division 2 - Södra Svealand
    597: None,    # Division 2 - Södra Götaland
    736: None,    # Elitettan
}

API_FOOTBALL_TO_FOTMOB_LEAGUE_MAPPING = {
    47: 39,       # Premier League
    48: 40,       # Championship
    108: 41,      # League One
    109: 42,      # League Two
    117: 43,      # National League
    132: 45,      # FA Cup
    53: 61,       # Ligue 1
    110: 62,      # Ligue 2
    54: 78,       # 1. Bundesliga -> Bundesliga
    146: 79,      # 2. Bundesliga
    57: 88,       # Eredivisie
    #61: 94,       # Liga Portugal -> Primeira Liga
    67: 113,      # Allsvenskan
    168: 114,     # Superettan
    55: 135,      # Serie A
    86: 136,      # Serie B
    87: 140,      # LaLiga -> La Liga
    140: 141,     # LaLiga2 -> Segunda División
    40: 144,      # First Division A -> Jupiler Pro League
    64: 179,      # Premiership
    123: 180,     # Championship
    124: 183,     # League One
    125: 184,     # League Two
    135: 197,     # Super League 1
    71: 203,      # Super Lig -> Süper Lig
    #9089: 549,    # Damallsvenskan (W) -> Damallsvenskan
    # FotMob Ettan (169) covers both API Ettan divisions:
    169: 563,     # primary: Ettan - Norra (also 564)
}

FOTMOBLEAGUE_EXTERNAL_ID_TO_CCODE = {
    39: "ENG",
    40: "ENG",
    41: "ENG",
    42: "ENG",
    43: "ENG",
    45: "ENG",
    61: "FRA",
    62: "FRA",
    78: "GER",
    79: "GER",
    88: "NED",
    94: "POR",
    113: "SWE",
    114: "SWE",
    135: "ITA",
    136: "ITA",
    140: "ESP",
    141: "ESP",
    144: "BEL",
    179: "SCO",
    180: "SCO",
    183: "SCO",
    184: "SCO",
    197: "GRE",
    203: "TUR",
    549: "SWE",
    563: "SWE",
    564: "SWE",
    592: "SWE",
    593: "SWE",
    594: "SWE",
    595: "SWE",
    597: "SWE",
    736: "SWE",
}

Outcome = Literal["1", "X", "2"]
OUTCOMES: tuple[Outcome, ...] = ("1", "X", "2")


class SignType(str, Enum):
    SPIK = "spik"
    HALV = "halv"
    HEL = "hel"


def fix_swedish_name(team_name):
    _name = swedish_to_ascii(team_name)
    if _name == 'BK Hacken':
        return 'Hacken'
    return _name

def swedish_to_ascii(text: str) -> str:
    """Convert Swedish å/ä/ö characters to ASCII-style equivalents."""
    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
        "Å": "A",
        "Ä": "A",
        "Ö": "O",
    }

    return "".join(replacements.get(char, char) for char in text)


def odds_to_probabilities(
    win_home: float,
    draw: float,
    win_away: float,
) -> dict[Outcome, float]:
    """Convert decimal 1X2 odds to normalized probabilities."""
    for label, odds in (("win_home", win_home), ("draw", draw), ("win_away", win_away)):
        if odds <= 0:
            raise ValueError(f"{label} odds must be positive, got {odds}")

    home = 1.0 / win_home
    draw_prob = 1.0 / draw
    away = 1.0 / win_away
    total = home + draw_prob + away

    return {
        "1": home / total,
        "X": draw_prob / total,
        "2": away / total,
    }


def ensure_unit_probabilities(
    market_probabilities: dict[str, float | None],
) -> dict[str, float | None]:
    """Return 1X2 values on the 0–1 scale. Reject mixed 0–1 / percentage inputs."""
    present_values = [
        value for value in market_probabilities.values() if value is not None
    ]
    if not present_values:
        return dict(market_probabilities)
    if any(value < 0 for value in present_values):
        raise ValueError("Market probabilities must be non-negative")

    all_unit_scale = all(value <= 1 for value in present_values)
    all_percent_scale = any(value > 1 for value in present_values) and all(
        value == 0 or value > 1 for value in present_values
    )
    if all_unit_scale:
        return dict(market_probabilities)
    if all_percent_scale:
        if any(value > 100 for value in present_values):
            raise ValueError("Percentage-style market probabilities must be <= 100")
        return {
            key: None if value is None else value / 100.0
            for key, value in market_probabilities.items()
        }
    raise ValueError("Mixed 0–1 and percentage market probability scales")

def probabilities_to_result(
    win_home: float,
    draw: float,
    win_away: float,
) -> dict[Outcome, float]:
    _all = [win_home, draw, win_away]
    if max(_all) == win_home:
        return '1'
    elif max(_all) == draw:
        return 'X'
    elif max(_all) == win_away:
        return '2'

def get_season_rev(season: str):
    seasons = {
        '2018': '2018/2019',
        '2019': '2019/2020',
        '2020': '2020/2021',
        '2021': '2021/2022',
        '2022': '2022/2023',
        '2023': '2023/2024',
        '2024': '2024/2025',
        '2025': '2025/2026',
    }
    return seasons[season]

def get_season(season: str):
    seasons = {
        '2022': '2223',
        '2023': '2324',
        '2024': '2425',
        '2025': '2526',
    }
    return seasons[season]



def strtr(s, repl):
  pattern = '|'.join(map(re.escape, sorted(repl, key=len, reverse=True)))
  return re.sub(pattern, lambda m: repl[m.group()], s)

def parse_swedish_decimal(value: str | float | int | None) -> float | None:
    """Parse Svenska Spel decimal strings like '3,10' or '1,00'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None

def sanitize_string(name: str):

  chars = {
    ',': '-',
    '.': '-',
    '/':'-',
    ' ':'-',
    '_':'-',
    'ª':'a',
    'º':'o',
    'À':'A',
    'Á':'A',
    'Â':'A',
    'Ã':'A',
    'Ä':'A',
    'Å':'A',
    'Æ':'AE',
    'Ç':'C',
    'È':'E',
    'É':'E',
    'Ê':'E',
    'Ë':'E',
    'Ì':'I',
    'Í':'I',
    'Î':'I',
    'Ï':'I',
    'Ð':'D',
    'Ñ':'N',
    'Ò':'O',
    'Ó':'O',
    'Ô':'O',
    'Õ':'O',
    'Ö':'O',
    'Ù':'U',
    'Ú':'U',
    'Û':'U',
    'Ü':'U',
    'Ý':'Y',
    'Þ':'TH',
    'ß':'s',
    'à':'a',
    'á':'a',
    'â':'a',
    'ã':'a',
    'ä':'a',
    'å':'a',
    'æ':'ae',
    'ç':'c',
    'è':'e',
    'é':'e',
    'ê':'e',
    'ë':'e',
    'ì':'i',
    'í':'i',
    'î':'i',
    'ï':'i',
    'ð':'d',
    'ñ':'n',
    'ò':'o',
    'ó':'o',
    'ô':'o',
    'õ':'o',
    'ö':'o',
    'ø':'o',
    'ù':'u',
    'ú':'u',
    'û':'u',
    'ü':'u',
    'ý':'y',
    'þ':'th',
    'ÿ':'y',
    'Ø':'O',
    'Ā':'A',
    'ā':'a',
    'Ă':'A',
    'ă':'a',
    'Ą':'A',
    'ą':'a',
    'Ć':'C',
    'ć':'c',
    'Ĉ':'C',
    'ĉ':'c',
    'Ċ':'C',
    'ċ':'c',
    'Č':'C',
    'č':'c',
    'Ď':'D',
    'ď':'d',
    'Đ':'D',
    'đ':'d',
    'Ē':'E',
    'ē':'e',
    'Ĕ':'E',
    'ĕ':'e',
    'Ė':'E',
    'ė':'e',
    'Ę':'E',
    'ę':'e',
    'Ě':'E',
    'ě':'e',
    'Ĝ':'G',
    'ĝ':'g',
    'Ğ':'G',
    'ğ':'g',
    'Ġ':'G',
    'ġ':'g',
    'Ģ':'G',
    'ģ':'g',
    'Ĥ':'H',
    'ĥ':'h',
    'Ħ':'H',
    'ħ':'h',
    'Ĩ':'I',
    'ĩ':'i',
    'Ī':'I',
    'ī':'i',
    'Ĭ':'I',
    'ĭ':'i',
    'Į':'I',
    'į':'i',
    'İ':'I',
    'ı':'i',
    'Ĳ':'IJ',
    'ĳ':'ij',
    'Ĵ':'J',
    'ĵ':'j',
    'Ķ':'K',
    'ķ':'k',
    'ĸ':'k',
    'Ĺ':'L',
    'ĺ':'l',
    'Ļ':'L',
    'ļ':'l',
    'Ľ':'L',
    'ľ':'l',
    'Ŀ':'L',
    'ŀ':'l',
    'Ł':'L',
    'ł':'l',
    'Ń':'N',
    'ń':'n',
    'Ņ':'N',
    'ņ':'n',
    'Ň':'N',
    'ň':'n',
    'ŉ':'n',
    'Ŋ':'N',
    'ŋ':'n',
    'Ō':'O',
    'ō':'o',
    'Ŏ':'O',
    'ŏ':'o',
    'Ő':'O',
    'ő':'o',
    'Œ':'OE',
    'œ':'oe',
    'Ŕ':'R',
    'ŕ':'r',
    'Ŗ':'R',
    'ŗ':'r',
    'Ř':'R',
    'ř':'r',
    'Ś':'S',
    'ś':'s',
    'Ŝ':'S',
    'ŝ':'s',
    'Ş':'S',
    'ş':'s',
    'Š':'S',
    'š':'s',
    'Ţ':'T',
    'ţ':'t',
    'Ť':'T',
    'ť':'t',
    'Ŧ':'T',
    'ŧ':'t',
    'Ũ':'U',
    'ũ':'u',
    'Ū':'U',
    'ū':'u',
    'Ŭ':'U',
    'ŭ':'u',
    'Ů':'U',
    'ů':'u',
    'Ű':'U',
    'ű':'u',
    'Ų':'U',
    'ų':'u',
    'Ŵ':'W',
    'ŵ':'w',
    'Ŷ':'Y',
    'ŷ':'y',
    'Ÿ':'Y',
    'Ź':'Z',
    'ź':'z',
    'Ż':'Z',
    'ż':'z',
    'Ž':'Z',
    'ž':'z',
    'ſ':'s',
    'Ș':'S',
    'ș':'s',
    'Ț':'T',
    'ț':'t',
    '€':'E',
    '£':'',
    'Ơ':'O',
    'ơ':'o',
    'Ư':'U',
    'ư':'u',
    'Ầ':'A',
    'ầ':'a',
    'Ằ':'A',
    'ằ':'a',
    'Ề':'E',
    'ề':'e',
    'Ồ':'O',
    'ồ':'o',
    'Ờ':'O',
    'ờ':'o',
    'Ừ':'U',
    'ừ':'u',
    'Ỳ':'Y',
    'ỳ':'y',
    'Ả':'A',
    'ả':'a',
    'Ẩ':'A',
    'ẩ':'a',
    'Ẳ':'A',
    'ẳ':'a',
    'Ẻ':'E',
    'ẻ':'e',
    'Ể':'E',
    'ể':'e',
    'Ỉ':'I',
    'ỉ':'i',
    'Ỏ':'O',
    'ỏ':'o',
    'Ổ':'O',
    'ổ':'o',
    'Ở':'O',
    'ở':'o',
    'Ủ':'U',
    'ủ':'u',
    'Ử':'U',
    'ử':'u',
    'Ỷ':'Y',
    'ỷ':'y',
    'Ẫ':'A',
    'ẫ':'a',
    'Ẵ':'A',
    'ẵ':'a',
    'Ẽ':'E',
    'ẽ':'e',
    'Ễ':'E',
    'ễ':'e',
    'Ỗ':'O',
    'ỗ':'o',
    'Ỡ':'O',
    'ỡ':'o',
    'Ữ':'U',
    'ữ':'u',
    'Ỹ':'Y',
    'ỹ':'y',
    'Ấ':'A',
    'ấ':'a',
    'Ắ':'A',
    'ắ':'a',
    'Ế':'E',
    'ế':'e',
    'Ố':'O',
    'ố':'o',
    'Ớ':'O',
    'ớ':'o',
    'Ứ':'U',
    'ứ':'u',
    'Ạ':'A',
    'ạ':'a',
    'Ậ':'A',
    'ậ':'a',
    'Ặ':'A',
    'ặ':'a',
    'Ẹ':'E',
    'ẹ':'e',
    'Ệ':'E',
    'ệ':'e',
    'Ị':'I',
    'ị':'i',
    'Ọ':'O',
    'ọ':'o',
    'Ộ':'O',
    'ộ':'o',
    'Ợ':'O',
    'ợ':'o',
    'Ụ':'U',
    'ụ':'u',
    'Ự':'U',
    'ự':'u',
    'Ỵ':'Y',
    'ỵ':'y',
    'ɑ':'a',
    'Ǖ':'U',
    'ǖ':'u',
    'Ǘ':'U',
    'ǘ':'u',
    'Ǎ':'A',
    'ǎ':'a',
    'Ǐ':'I',
    'ǐ':'i',
    'Ǒ':'O',
    'ǒ':'o',
    'Ǔ':'U',
    'ǔ':'u',
    'Ǚ':'U',
    'ǚ':'u',
    'Ǜ':'U',
    'ǜ':'u',
    '&':'-',
    ' - ':'-',
    ')':'-',
    '(':'-',
  }
  name = strtr(name, chars).lower()
  name = name.rstrip('-')
  return name
