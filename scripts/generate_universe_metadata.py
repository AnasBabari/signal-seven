"""Deterministic generator for the unified 286-ticker tri-exchange metadata catalog.

Produces:
1. ``backend/data/universe_metadata.json`` — authoritative 286-ticker metadata catalog
   (98 NASDAQ / 97 NYSE / 91 LSE; 195 US non-``.L`` with 10-digit zero-padded SEC CIKs,
   91 LSE ``.L`` with ``cik=None``, M2 quote-unit overrides for ``CPG.L`` and ``IHG.L``
   in ``USD``/``$``, standardized 11-sector GICS classification, and sector crosswalk).
2. ``frontend/src/universe.js`` — lightweight frontend universe module preserving all
   existing exports (``NASDAQ_TICKERS``, ``NYSE_TICKERS``, ``LSE_TICKERS``,
   ``ALL_VALID_TICKERS``, ``ALL_TICKERS_SET``) plus ``SUPPORTED_TICKERS``,
   ``SUPPORTED_TICKER_SET``, ``TICKER_METADATA``, ``normalizeTicker``,
   ``isSupportedTicker``, and ``getTickerMetadata``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from market_data.quote_units import LSE_QUOTE_UNIT_OVERRIDES  # noqa: E402

GENERATED_DATE = "2026-11-23"
SCHEMA_VERSION = "universe-metadata-v1"

GICS_SECTORS: tuple[str, ...] = (
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
)

SECTOR_CROSSWALK: dict[str, str] = {
    "Technology": "Information Technology",
    "Information Technology": "Information Technology",
    "Tech": "Information Technology",
    "Healthcare": "Health Care",
    "Health Care": "Health Care",
    "Financial Services": "Financials",
    "Financial": "Financials",
    "Financials": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Staples": "Consumer Staples",
    "Communication Services": "Communication Services",
    "Telecommunications": "Communication Services",
    "Industrials": "Industrials",
    "Industrial": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Basic Materials": "Materials",
    "Materials": "Materials",
}

CIK_OVERRIDES: dict[str, str] = {
    "WBD": "0001437107",  # Warner Bros. Discovery, Inc.
    "GOOGL": "0001652044",  # Alphabet Inc. (Class A share class)
}

NASDAQ_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "COST",
    "QCOM",
    "AVGO",
    "NFLX",
    "ADBE",
    "TXN",
    "INTC",
    "CMCSA",
    "PEP",
    "CSCO",
    "INTU",
    "AMAT",
    "PYPL",
    "BKNG",
    "ISRG",
    "MDLZ",
    "GILD",
    "REGN",
    "ADI",
    "VRTX",
    "LRCX",
    "PANW",
    "SNPS",
    "KLAC",
    "CDNS",
    "CHTR",
    "MAR",
    "ORLY",
    "NXPI",
    "FTNT",
    "CTAS",
    "PCAR",
    "KDP",
    "PAYX",
    "MNST",
    "MCHP",
    "ROST",
    "AEP",
    "KHC",
    "ODFL",
    "FAST",
    "IDXX",
    "EXC",
    "LULU",
    "VRSK",
    "CSX",
    "GEHC",
    "BIIB",
    "DXCM",
    "XEL",
    "FANG",
    "TEAM",
    "MRVL",
    "ON",
    "BKR",
    "WBD",
    "DLTR",
    "ILMN",
    "ALGN",
    "CEG",
    "CPRT",
    "SBUX",
    "HON",
    "ASML",
    "AZN",
    "PDD",
    "MELI",
    "CRWD",
    "DDOG",
    "ZS",
    "WDAY",
    "ABNB",
    "MRNA",
    "DASH",
    "TTD",
    "MDB",
    "SMCI",
    "ARM",
    "APP",
    "PLTR",
    "ROP",
    "TTWO",
    "CDW",
    "GFS",
    "ANET",
    "CCEP",
    "AXON",
    "MSTR",
    "LIN",
    "CTSH",
)

NYSE_TICKERS: tuple[str, ...] = (
    "JPM",
    "XOM",
    "WMT",
    "JNJ",
    "CAT",
    "KO",
    "NEE",
    "DIS",
    "BAC",
    "GE",
    "UNH",
    "PG",
    "V",
    "HD",
    "CVX",
    "MA",
    "PFE",
    "MRK",
    "ABT",
    "ORCL",
    "LLY",
    "MCD",
    "TMO",
    "NKE",
    "IBM",
    "GS",
    "MS",
    "RTX",
    "UNP",
    "BMY",
    "COP",
    "PM",
    "LOW",
    "UPS",
    "MSI",
    "DE",
    "SCHW",
    "C",
    "AXP",
    "BLK",
    "PGR",
    "CB",
    "AON",
    "MET",
    "PRU",
    "AIG",
    "TRV",
    "ALL",
    "EOG",
    "SLB",
    "MPC",
    "PSX",
    "VLO",
    "OXY",
    "DVN",
    "KMI",
    "WMB",
    "ET",
    "MDT",
    "SYK",
    "BSX",
    "EW",
    "BAX",
    "BDX",
    "ZTS",
    "CI",
    "ELV",
    "HUM",
    "CVS",
    "FDX",
    "NSC",
    "WM",
    "RSG",
    "EMR",
    "ETN",
    "ITW",
    "PH",
    "CMI",
    "TT",
    "JCI",
    "CARR",
    "OTIS",
    "GD",
    "NOC",
    "LHX",
    "LMT",
    "BA",
    "HMC",
    "TM",
    "SONY",
    "NVO",
    "BABA",
    "NVS",
    "SAP",
    "TSM",
    "DEO",
    "BAM",
)

LSE_TICKERS: tuple[str, ...] = (
    "SHEL.L",
    "AZN.L",
    "HSBA.L",
    "BP.L",
    "ULVR.L",
    "GSK.L",
    "RIO.L",
    "BATS.L",
    "BARC.L",
    "DGE.L",
    "REL.L",
    "LSEG.L",
    "LLOY.L",
    "NG.L",
    "EXPN.L",
    "VOD.L",
    "GLEN.L",
    "AAL.L",
    "PRU.L",
    "NWG.L",
    "STAN.L",
    "BA.L",
    "CPG.L",
    "IMB.L",
    "AV.L",
    "SSE.L",
    "RR.L",
    "TSCO.L",
    "ABF.L",
    "ADM.L",
    "ANTO.L",
    "INF.L",
    "MNDI.L",
    "WPP.L",
    "HLN.L",
    "SDR.L",
    "LAND.L",
    "BLND.L",
    "UU.L",
    "SVT.L",
    "WTB.L",
    "AUTO.L",
    "ENT.L",
    "JD.L",
    "KGF.L",
    "MKS.L",
    "NXT.L",
    "BME.L",
    "OCDO.L",
    "SMIN.L",
    "SMT.L",
    "FDM.L",
    "PSN.L",
    "TW.L",
    "CRDA.L",
    "JMAT.L",
    "HLMA.L",
    "SPX.L",
    "WEIR.L",
    "IMI.L",
    "SN.L",
    "SGE.L",
    "RS1.L",
    "BEZ.L",
    "HIK.L",
    "IHG.L",
    "LGEN.L",
    "STJ.L",
    "UTG.L",
    "ITRK.L",
    "BKG.L",
    "SGRO.L",
    "PSON.L",
    "DCC.L",
    "SBRY.L",
    "FRAS.L",
    "RMV.L",
    "WISE.L",
    "EMG.L",
    "MONY.L",
    "EDV.L",
    "GAW.L",
    "IGG.L",
    "VTY.L",
    "WIZZ.L",
    "GNS.L",
    "BOY.L",
    "ITV.L",
    "GFTU.L",
    "TRN.L",
    "EZJ.L",
)

# Per-symbol curated table: (name, gics_sector, gics_industry, sec_cik_10digit_or_None)
# Preserves exact company names from backend/services/simple_forecast.py and quote_units.py.
CURATED_CATALOG: dict[str, tuple[str, str, str, str | None]] = {
    # --- NASDAQ (98) ---
    "AAPL": (
        "Apple Inc.",
        "Information Technology",
        "Technology Hardware, Storage & Peripherals",
        "0000320193",
    ),
    "MSFT": ("Microsoft Corp.", "Information Technology", "Systems Software", "0000789019"),
    "NVDA": ("NVIDIA Corp.", "Information Technology", "Semiconductors", "0001045810"),
    "GOOGL": (
        "Alphabet Inc.",
        "Communication Services",
        "Interactive Media & Services",
        "0001652044",
    ),
    "AMZN": ("Amazon.com Inc.", "Consumer Discretionary", "Broadline Retail", "0001018724"),
    "META": (
        "Meta Platforms Inc.",
        "Communication Services",
        "Interactive Media & Services",
        "0001326801",
    ),
    "TSLA": ("Tesla Inc.", "Consumer Discretionary", "Automobile Manufacturers", "0001318605"),
    "AMD": ("Advanced Micro Devices", "Information Technology", "Semiconductors", "0000002488"),
    "COST": (
        "Costco Wholesale Corp.",
        "Consumer Staples",
        "Consumer Staples Merchandise Retail",
        "0000909832",
    ),
    "QCOM": ("Qualcomm Inc.", "Information Technology", "Semiconductors", "0000804328"),
    "AVGO": ("Broadcom Inc.", "Information Technology", "Semiconductors", "0001730168"),
    "NFLX": ("Netflix Inc.", "Communication Services", "Movies & Entertainment", "0001065280"),
    "ADBE": ("Adobe Inc.", "Information Technology", "Application Software", "0000796343"),
    "TXN": ("Texas Instruments Inc.", "Information Technology", "Semiconductors", "0000097476"),
    "INTC": ("Intel Corp.", "Information Technology", "Semiconductors", "0000050863"),
    "CMCSA": ("Comcast Corp.", "Communication Services", "Cable & Satellite", "0001166691"),
    "PEP": (
        "PepsiCo Inc.",
        "Consumer Staples",
        "Soft Drinks & Non-alcoholic Beverages",
        "0000077476",
    ),
    "CSCO": (
        "Cisco Systems Inc.",
        "Information Technology",
        "Communications Equipment",
        "0000858877",
    ),
    "INTU": ("Intuit Inc.", "Information Technology", "Application Software", "0000896878"),
    "AMAT": (
        "Applied Materials Inc.",
        "Information Technology",
        "Semiconductor Materials & Equipment",
        "0000006951",
    ),
    "PYPL": (
        "PayPal Holdings Inc.",
        "Financials",
        "Transaction & Payment Processing Services",
        "0001633917",
    ),
    "BKNG": (
        "Booking Holdings Inc.",
        "Consumer Discretionary",
        "Hotels, Resorts & Cruise Lines",
        "0001075531",
    ),
    "ISRG": ("Intuitive Surgical Inc.", "Health Care", "Health Care Equipment", "0001035267"),
    "MDLZ": (
        "Mondelez International Inc.",
        "Consumer Staples",
        "Packaged Foods & Meats",
        "0001103982",
    ),
    "GILD": ("Gilead Sciences Inc.", "Health Care", "Biotechnology", "0000882095"),
    "REGN": ("Regeneron Pharmaceuticals Inc.", "Health Care", "Biotechnology", "0000872589"),
    "ADI": ("Analog Devices Inc.", "Information Technology", "Semiconductors", "0000006281"),
    "VRTX": ("Vertex Pharmaceuticals Inc.", "Health Care", "Biotechnology", "0000875320"),
    "LRCX": (
        "Lam Research Corp.",
        "Information Technology",
        "Semiconductor Materials & Equipment",
        "0000707549",
    ),
    "PANW": ("Palo Alto Networks Inc.", "Information Technology", "Systems Software", "0001327567"),
    "SNPS": ("Synopsys Inc.", "Information Technology", "Application Software", "0000883241"),
    "KLAC": (
        "KLA Corp.",
        "Information Technology",
        "Semiconductor Materials & Equipment",
        "0000319201",
    ),
    "CDNS": (
        "Cadence Design Systems Inc.",
        "Information Technology",
        "Application Software",
        "0000813672",
    ),
    "CHTR": (
        "Charter Communications Inc.",
        "Communication Services",
        "Cable & Satellite",
        "0001091667",
    ),
    "MAR": (
        "Marriott International Inc.",
        "Consumer Discretionary",
        "Hotels, Resorts & Cruise Lines",
        "0001048286",
    ),
    "ORLY": (
        "O'Reilly Automotive Inc.",
        "Consumer Discretionary",
        "Automotive Retail",
        "0000898173",
    ),
    "NXPI": ("NXP Semiconductors N.V.", "Information Technology", "Semiconductors", "0001413447"),
    "FTNT": ("Fortinet Inc.", "Information Technology", "Systems Software", "0001262039"),
    "CTAS": ("Cintas Corp.", "Industrials", "Diversified Support Services", "0000723254"),
    "PCAR": (
        "PACCAR Inc.",
        "Industrials",
        "Construction Machinery & Heavy Transportation Equipment",
        "0000075362",
    ),
    "KDP": (
        "Keurig Dr Pepper Inc.",
        "Consumer Staples",
        "Soft Drinks & Non-alcoholic Beverages",
        "0001418135",
    ),
    "PAYX": ("Paychex Inc.", "Industrials", "Human Resource & Employment Services", "0000723531"),
    "MNST": (
        "Monster Beverage Corp.",
        "Consumer Staples",
        "Soft Drinks & Non-alcoholic Beverages",
        "0000865752",
    ),
    "MCHP": ("Microchip Technology Inc.", "Information Technology", "Semiconductors", "0000827054"),
    "ROST": ("Ross Stores Inc.", "Consumer Discretionary", "Apparel Retail", "0000745732"),
    "AEP": ("American Electric Power Co. Inc.", "Utilities", "Electric Utilities", "0000004904"),
    "KHC": ("The Kraft Heinz Co.", "Consumer Staples", "Packaged Foods & Meats", "0001637459"),
    "ODFL": (
        "Old Dominion Freight Line Inc.",
        "Industrials",
        "Cargo Ground Transportation",
        "0000878927",
    ),
    "FAST": ("Fastenal Co.", "Industrials", "Trading Companies & Distributors", "0000815556"),
    "IDXX": ("IDEXX Laboratories Inc.", "Health Care", "Health Care Equipment", "0000874716"),
    "EXC": ("Exelon Corp.", "Utilities", "Electric Utilities", "0001109357"),
    "LULU": (
        "lululemon athletica inc.",
        "Consumer Discretionary",
        "Apparel, Accessories & Luxury Goods",
        "0001397187",
    ),
    "VRSK": (
        "Verisk Analytics Inc.",
        "Industrials",
        "Research & Consulting Services",
        "0001442145",
    ),
    "CSX": ("CSX Corp.", "Industrials", "Rail Transportation", "0000277948"),
    "GEHC": (
        "GE HealthCare Technologies Inc.",
        "Health Care",
        "Health Care Equipment",
        "0001932393",
    ),
    "BIIB": ("Biogen Inc.", "Health Care", "Biotechnology", "0000875045"),
    "DXCM": ("DexCom Inc.", "Health Care", "Health Care Equipment", "0001093557"),
    "XEL": ("Xcel Energy Inc.", "Utilities", "Multi-Utilities", "0000072903"),
    "FANG": (
        "Diamondback Energy Inc.",
        "Energy",
        "Oil & Gas Exploration & Production",
        "0001539838",
    ),
    "TEAM": ("Atlassian Corp.", "Information Technology", "Application Software", "0001650372"),
    "MRVL": ("Marvell Technology Inc.", "Information Technology", "Semiconductors", "0001835632"),
    "ON": ("ON Semiconductor Corp.", "Information Technology", "Semiconductors", "0001097864"),
    "BKR": ("Baker Hughes Co.", "Energy", "Oil & Gas Equipment & Services", "0001701605"),
    "WBD": (
        "Warner Bros. Discovery Inc.",
        "Communication Services",
        "Movies & Entertainment",
        "0001437107",
    ),
    "DLTR": (
        "Dollar Tree Inc.",
        "Consumer Staples",
        "Consumer Staples Merchandise Retail",
        "0000935703",
    ),
    "ILMN": ("Illumina Inc.", "Health Care", "Life Sciences Tools & Services", "0001110803"),
    "ALGN": ("Align Technology Inc.", "Health Care", "Health Care Supplies", "0001097149"),
    "CEG": ("Constellation Energy Corp.", "Utilities", "Electric Utilities", "0001868275"),
    "CPRT": ("Copart Inc.", "Industrials", "Diversified Support Services", "0000900075"),
    "SBUX": ("Starbucks Corp.", "Consumer Discretionary", "Restaurants", "0000829224"),
    "HON": (
        "Honeywell International Inc.",
        "Industrials",
        "Industrial Conglomerates",
        "0000773840",
    ),
    "ASML": (
        "ASML Holding N.V.",
        "Information Technology",
        "Semiconductor Materials & Equipment",
        "0000937966",
    ),
    "AZN": ("AstraZeneca PLC (ADR)", "Health Care", "Pharmaceuticals", "0000901832"),
    "PDD": ("PDD Holdings Inc.", "Consumer Discretionary", "Broadline Retail", "0001737806"),
    "MELI": ("MercadoLibre Inc.", "Consumer Discretionary", "Broadline Retail", "0001099590"),
    "CRWD": (
        "CrowdStrike Holdings Inc.",
        "Information Technology",
        "Systems Software",
        "0001535527",
    ),
    "DDOG": ("Datadog Inc.", "Information Technology", "Application Software", "0001561550"),
    "ZS": ("Zscaler Inc.", "Information Technology", "Systems Software", "0001713683"),
    "WDAY": ("Workday Inc.", "Information Technology", "Application Software", "0001327811"),
    "ABNB": (
        "Airbnb Inc.",
        "Consumer Discretionary",
        "Hotels, Resorts & Cruise Lines",
        "0001559720",
    ),
    "MRNA": ("Moderna Inc.", "Health Care", "Biotechnology", "0001682852"),
    "DASH": (
        "DoorDash Inc.",
        "Consumer Discretionary",
        "Specialized Consumer Services",
        "0001792789",
    ),
    "TTD": ("The Trade Desk Inc.", "Communication Services", "Advertising", "0001671933"),
    "MDB": ("MongoDB Inc.", "Information Technology", "Systems Software", "0001441816"),
    "SMCI": (
        "Super Micro Computer Inc.",
        "Information Technology",
        "Technology Hardware, Storage & Peripherals",
        "0001375365",
    ),
    "ARM": ("Arm Holdings plc", "Information Technology", "Semiconductors", "0001973239"),
    "APP": ("AppLovin Corp.", "Information Technology", "Application Software", "0001751008"),
    "PLTR": (
        "Palantir Technologies Inc.",
        "Information Technology",
        "Application Software",
        "0001321655",
    ),
    "ROP": (
        "Roper Technologies Inc.",
        "Information Technology",
        "Application Software",
        "0000882835",
    ),
    "TTWO": (
        "Take-Two Interactive Software Inc.",
        "Communication Services",
        "Interactive Home Entertainment",
        "0000946581",
    ),
    "CDW": ("CDW Corp.", "Information Technology", "Technology Distributors", "0001402057"),
    "GFS": ("GLOBALFOUNDRIES Inc.", "Information Technology", "Semiconductors", "0001709048"),
    "ANET": (
        "Arista Networks Inc.",
        "Information Technology",
        "Communications Equipment",
        "0001596532",
    ),
    "CCEP": (
        "Coca-Cola Europacific Partners plc",
        "Consumer Staples",
        "Soft Drinks & Non-alcoholic Beverages",
        "0001650107",
    ),
    "AXON": ("Axon Enterprise Inc.", "Industrials", "Aerospace & Defense", "0001069183"),
    "MSTR": ("MicroStrategy Inc.", "Information Technology", "Application Software", "0001050446"),
    "LIN": ("Linde plc", "Materials", "Industrial Gases", "0001707925"),
    "CTSH": (
        "Cognizant Technology Solutions Corp.",
        "Information Technology",
        "IT Consulting & Other Services",
        "0001058290",
    ),
    # --- NYSE (97) ---
    "JPM": ("JPMorgan Chase & Co.", "Financials", "Diversified Banks", "0000019617"),
    "XOM": ("Exxon Mobil Corp.", "Energy", "Integrated Oil & Gas", "0002115436"),
    "WMT": (
        "Walmart Inc.",
        "Consumer Staples",
        "Consumer Staples Merchandise Retail",
        "0000104169",
    ),
    "JNJ": ("Johnson & Johnson", "Health Care", "Pharmaceuticals", "0000200406"),
    "CAT": (
        "Caterpillar Inc.",
        "Industrials",
        "Construction Machinery & Heavy Transportation Equipment",
        "0000018230",
    ),
    "KO": (
        "The Coca-Cola Company",
        "Consumer Staples",
        "Soft Drinks & Non-alcoholic Beverages",
        "0000021344",
    ),
    "NEE": ("NextEra Energy Inc.", "Utilities", "Electric Utilities", "0000753308"),
    "DIS": (
        "The Walt Disney Company",
        "Communication Services",
        "Movies & Entertainment",
        "0001744489",
    ),
    "BAC": ("Bank of America Corp.", "Financials", "Diversified Banks", "0000070858"),
    "GE": ("GE Aerospace", "Industrials", "Aerospace & Defense", "0000040545"),
    "UNH": ("UnitedHealth Group Inc.", "Health Care", "Managed Health Care", "0000731766"),
    "PG": ("The Procter & Gamble Co.", "Consumer Staples", "Household Products", "0000080424"),
    "V": ("Visa Inc.", "Financials", "Transaction & Payment Processing Services", "0001403161"),
    "HD": (
        "The Home Depot Inc.",
        "Consumer Discretionary",
        "Home Improvement Retail",
        "0000354950",
    ),
    "CVX": ("Chevron Corp.", "Energy", "Integrated Oil & Gas", "0000093410"),
    "MA": (
        "Mastercard Inc.",
        "Financials",
        "Transaction & Payment Processing Services",
        "0001141391",
    ),
    "PFE": ("Pfizer Inc.", "Health Care", "Pharmaceuticals", "0000078003"),
    "MRK": ("Merck & Co. Inc.", "Health Care", "Pharmaceuticals", "0000310158"),
    "ABT": ("Abbott Laboratories", "Health Care", "Health Care Equipment", "0000001800"),
    "ORCL": ("Oracle Corp.", "Information Technology", "Systems Software", "0001341439"),
    "LLY": ("Eli Lilly and Co.", "Health Care", "Pharmaceuticals", "0000059478"),
    "MCD": ("McDonald's Corp.", "Consumer Discretionary", "Restaurants", "0000063908"),
    "TMO": (
        "Thermo Fisher Scientific Inc.",
        "Health Care",
        "Life Sciences Tools & Services",
        "0000097745",
    ),
    "NKE": ("NIKE Inc.", "Consumer Discretionary", "Footwear", "0000320187"),
    "IBM": (
        "International Business Machines Corp.",
        "Information Technology",
        "IT Consulting & Other Services",
        "0000051143",
    ),
    "GS": (
        "The Goldman Sachs Group Inc.",
        "Financials",
        "Investment Banking & Brokerage",
        "0000886982",
    ),
    "MS": ("Morgan Stanley", "Financials", "Investment Banking & Brokerage", "0000895421"),
    "RTX": ("RTX Corp.", "Industrials", "Aerospace & Defense", "0000101829"),
    "UNP": ("Union Pacific Corp.", "Industrials", "Rail Transportation", "0000100885"),
    "BMY": ("Bristol-Myers Squibb Co.", "Health Care", "Pharmaceuticals", "0000014272"),
    "COP": ("ConocoPhillips", "Energy", "Oil & Gas Exploration & Production", "0001163165"),
    "PM": ("Philip Morris International Inc.", "Consumer Staples", "Tobacco", "0001413329"),
    "LOW": (
        "Lowe's Companies Inc.",
        "Consumer Discretionary",
        "Home Improvement Retail",
        "0000060667",
    ),
    "UPS": ("United Parcel Service Inc.", "Industrials", "Air Freight & Logistics", "0001090727"),
    "MSI": (
        "Motorola Solutions Inc.",
        "Information Technology",
        "Communications Equipment",
        "0000068505",
    ),
    "DE": ("Deere & Co.", "Industrials", "Agricultural & Farm Machinery", "0000315189"),
    "SCHW": (
        "The Charles Schwab Corp.",
        "Financials",
        "Investment Banking & Brokerage",
        "0000316709",
    ),
    "C": ("Citigroup Inc.", "Financials", "Diversified Banks", "0000831001"),
    "AXP": ("American Express Co.", "Financials", "Consumer Finance", "0000004962"),
    "BLK": ("BlackRock Inc.", "Financials", "Asset Management & Custody Banks", "0002012383"),
    "PGR": ("The Progressive Corp.", "Financials", "Property & Casualty Insurance", "0000080661"),
    "CB": ("Chubb Ltd.", "Financials", "Property & Casualty Insurance", "0000896159"),
    "AON": ("Aon plc", "Financials", "Insurance Brokers", "0000315293"),
    "MET": ("MetLife Inc.", "Financials", "Life & Health Insurance", "0001099219"),
    "PRU": ("Prudential Financial Inc.", "Financials", "Life & Health Insurance", "0001137774"),
    "AIG": (
        "American International Group Inc.",
        "Financials",
        "Property & Casualty Insurance",
        "0000005272",
    ),
    "TRV": (
        "The Travelers Companies Inc.",
        "Financials",
        "Property & Casualty Insurance",
        "0000086312",
    ),
    "ALL": ("The Allstate Corp.", "Financials", "Property & Casualty Insurance", "0000899051"),
    "EOG": ("EOG Resources Inc.", "Energy", "Oil & Gas Exploration & Production", "0000821189"),
    "SLB": ("SLB N.V.", "Energy", "Oil & Gas Equipment & Services", "0000087347"),
    "MPC": ("Marathon Petroleum Corp.", "Energy", "Oil & Gas Refining & Marketing", "0001510295"),
    "PSX": ("Phillips 66", "Energy", "Oil & Gas Refining & Marketing", "0001534701"),
    "VLO": ("Valero Energy Corp.", "Energy", "Oil & Gas Refining & Marketing", "0001035002"),
    "OXY": (
        "Occidental Petroleum Corp.",
        "Energy",
        "Oil & Gas Exploration & Production",
        "0000797468",
    ),
    "DVN": ("Devon Energy Corp.", "Energy", "Oil & Gas Exploration & Production", "0001090012"),
    "KMI": ("Kinder Morgan Inc.", "Energy", "Oil & Gas Storage & Transportation", "0001506307"),
    "WMB": (
        "The Williams Companies Inc.",
        "Energy",
        "Oil & Gas Storage & Transportation",
        "0000107263",
    ),
    "ET": ("Energy Transfer LP", "Energy", "Oil & Gas Storage & Transportation", "0001276187"),
    "MDT": ("Medtronic plc", "Health Care", "Health Care Equipment", "0001613103"),
    "SYK": ("Stryker Corp.", "Health Care", "Health Care Equipment", "0000310764"),
    "BSX": ("Boston Scientific Corp.", "Health Care", "Health Care Equipment", "0000885725"),
    "EW": ("Edwards Lifesciences Corp.", "Health Care", "Health Care Equipment", "0001099800"),
    "BAX": ("Baxter International Inc.", "Health Care", "Health Care Equipment", "0000010456"),
    "BDX": ("Becton, Dickinson and Co.", "Health Care", "Health Care Equipment", "0000010795"),
    "ZTS": ("Zoetis Inc.", "Health Care", "Pharmaceuticals", "0001555280"),
    "CI": ("The Cigna Group", "Health Care", "Health Care Services", "0001739940"),
    "ELV": ("Elevance Health Inc.", "Health Care", "Managed Health Care", "0001156039"),
    "HUM": ("Humana Inc.", "Health Care", "Managed Health Care", "0000049071"),
    "CVS": ("CVS Health Corp.", "Health Care", "Health Care Services", "0000064803"),
    "FDX": ("FedEx Corp.", "Industrials", "Air Freight & Logistics", "0001048911"),
    "NSC": ("Norfolk Southern Corp.", "Industrials", "Rail Transportation", "0000702165"),
    "WM": (
        "Waste Management Inc.",
        "Industrials",
        "Environmental & Facilities Services",
        "0000823768",
    ),
    "RSG": (
        "Republic Services Inc.",
        "Industrials",
        "Environmental & Facilities Services",
        "0001060391",
    ),
    "EMR": (
        "Emerson Electric Co.",
        "Industrials",
        "Electrical Components & Equipment",
        "0000032604",
    ),
    "ETN": ("Eaton Corp. plc", "Industrials", "Electrical Components & Equipment", "0001551182"),
    "ITW": (
        "Illinois Tool Works Inc.",
        "Industrials",
        "Industrial Machinery & Supplies & Components",
        "0000049826",
    ),
    "PH": (
        "Parker-Hannifin Corp.",
        "Industrials",
        "Industrial Machinery & Supplies & Components",
        "0000076334",
    ),
    "CMI": (
        "Cummins Inc.",
        "Industrials",
        "Construction Machinery & Heavy Transportation Equipment",
        "0000026172",
    ),
    "TT": ("Trane Technologies plc", "Industrials", "Building Products", "0001466258"),
    "JCI": ("Johnson Controls International plc", "Industrials", "Building Products", "0000833444"),
    "CARR": ("Carrier Global Corp.", "Industrials", "Building Products", "0001783180"),
    "OTIS": (
        "Otis Worldwide Corp.",
        "Industrials",
        "Industrial Machinery & Supplies & Components",
        "0001781335",
    ),
    "GD": ("General Dynamics Corp.", "Industrials", "Aerospace & Defense", "0000040533"),
    "NOC": ("Northrop Grumman Corp.", "Industrials", "Aerospace & Defense", "0001133421"),
    "LHX": ("L3Harris Technologies Inc.", "Industrials", "Aerospace & Defense", "0000202058"),
    "LMT": ("Lockheed Martin Corp.", "Industrials", "Aerospace & Defense", "0000936468"),
    "BA": ("The Boeing Co.", "Industrials", "Aerospace & Defense", "0000012927"),
    "HMC": (
        "Honda Motor Co. Ltd.",
        "Consumer Discretionary",
        "Automobile Manufacturers",
        "0000715153",
    ),
    "TM": (
        "Toyota Motor Corp.",
        "Consumer Discretionary",
        "Automobile Manufacturers",
        "0001094517",
    ),
    "SONY": ("Sony Group Corp.", "Consumer Discretionary", "Consumer Electronics", "0000313838"),
    "NVO": ("Novo Nordisk A/S", "Health Care", "Pharmaceuticals", "0000353278"),
    "BABA": (
        "Alibaba Group Holding Ltd.",
        "Consumer Discretionary",
        "Broadline Retail",
        "0001577552",
    ),
    "NVS": ("Novartis AG", "Health Care", "Pharmaceuticals", "0001114448"),
    "SAP": ("SAP SE", "Information Technology", "Application Software", "0001000184"),
    "TSM": (
        "Taiwan Semiconductor Manufacturing Co. Ltd.",
        "Information Technology",
        "Semiconductors",
        "0001046179",
    ),
    "DEO": ("Diageo plc (ADR)", "Consumer Staples", "Distillers & Vintners", "0000835403"),
    "BAM": (
        "Brookfield Asset Management Ltd.",
        "Financials",
        "Asset Management & Custody Banks",
        "0001937926",
    ),
    # --- LSE (91) ---
    "SHEL.L": ("Shell plc", "Energy", "Integrated Oil & Gas", None),
    "AZN.L": ("AstraZeneca PLC", "Health Care", "Pharmaceuticals", None),
    "HSBA.L": ("HSBC Holdings plc", "Financials", "Diversified Banks", None),
    "BP.L": ("BP p.l.c.", "Energy", "Integrated Oil & Gas", None),
    "ULVR.L": ("Unilever PLC", "Consumer Staples", "Personal Care Products", None),
    "GSK.L": ("GSK plc", "Health Care", "Pharmaceuticals", None),
    "RIO.L": ("Rio Tinto plc", "Materials", "Diversified Metals & Mining", None),
    "BATS.L": ("British American Tobacco p.l.c.", "Consumer Staples", "Tobacco", None),
    "BARC.L": ("Barclays PLC", "Financials", "Diversified Banks", None),
    "DGE.L": ("Diageo plc", "Consumer Staples", "Distillers & Vintners", None),
    "REL.L": ("RELX PLC", "Industrials", "Research & Consulting Services", None),
    "LSEG.L": ("London Stock Exchange Group plc", "Financials", "Financial Exchanges & Data", None),
    "LLOY.L": ("Lloyds Banking Group plc", "Financials", "Diversified Banks", None),
    "NG.L": ("National Grid plc", "Utilities", "Multi-Utilities", None),
    "EXPN.L": ("Experian plc", "Industrials", "Research & Consulting Services", None),
    "VOD.L": (
        "Vodafone Group Plc",
        "Communication Services",
        "Wireless Telecommunication Services",
        None,
    ),
    "GLEN.L": ("Glencore plc", "Materials", "Diversified Metals & Mining", None),
    "AAL.L": ("Anglo American plc", "Materials", "Diversified Metals & Mining", None),
    "PRU.L": ("Prudential plc", "Financials", "Life & Health Insurance", None),
    "NWG.L": ("NatWest Group plc", "Financials", "Diversified Banks", None),
    "STAN.L": ("Standard Chartered PLC", "Financials", "Diversified Banks", None),
    "BA.L": ("BAE Systems plc", "Industrials", "Aerospace & Defense", None),
    "CPG.L": ("Compass Group plc", "Consumer Discretionary", "Restaurants", None),
    "IMB.L": ("Imperial Brands PLC", "Consumer Staples", "Tobacco", None),
    "AV.L": ("Aviva plc", "Financials", "Multi-line Insurance", None),
    "SSE.L": ("SSE plc", "Utilities", "Electric Utilities", None),
    "RR.L": ("Rolls-Royce Holdings plc", "Industrials", "Aerospace & Defense", None),
    "TSCO.L": ("Tesco PLC", "Consumer Staples", "Food Retail", None),
    "ABF.L": ("Associated British Foods plc", "Consumer Staples", "Packaged Foods & Meats", None),
    "ADM.L": ("Admiral Group plc", "Financials", "Property & Casualty Insurance", None),
    "ANTO.L": ("Antofagasta plc", "Materials", "Copper", None),
    "INF.L": ("Informa plc", "Communication Services", "Publishing", None),
    "MNDI.L": ("Mondi plc", "Materials", "Paper & Plastic Packaging Products & Materials", None),
    "WPP.L": ("WPP plc", "Communication Services", "Advertising", None),
    "HLN.L": ("Haleon plc", "Consumer Staples", "Personal Care Products", None),
    "SDR.L": ("Schroders plc", "Financials", "Asset Management & Custody Banks", None),
    "LAND.L": ("Land Securities Group plc", "Real Estate", "Diversified REITs", None),
    "BLND.L": ("British Land Company plc", "Real Estate", "Diversified REITs", None),
    "UU.L": ("United Utilities Group PLC", "Utilities", "Water Utilities", None),
    "SVT.L": ("Severn Trent Plc", "Utilities", "Water Utilities", None),
    "WTB.L": ("Whitbread PLC", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", None),
    "AUTO.L": (
        "Auto Trader Group plc",
        "Communication Services",
        "Interactive Media & Services",
        None,
    ),
    "ENT.L": ("Entain plc", "Consumer Discretionary", "Casinos & Gaming", None),
    "JD.L": ("JD Sports Fashion plc", "Consumer Discretionary", "Apparel Retail", None),
    "KGF.L": ("Kingfisher plc", "Consumer Discretionary", "Home Improvement Retail", None),
    "MKS.L": ("Marks and Spencer Group plc", "Consumer Staples", "Food Retail", None),
    "NXT.L": ("Next plc", "Consumer Discretionary", "Apparel Retail", None),
    "BME.L": ("B&M European Value Retail S.A.", "Consumer Discretionary", "Broadline Retail", None),
    "OCDO.L": ("Ocado Group plc", "Consumer Staples", "Food Retail", None),
    "SMIN.L": ("Smiths Group plc", "Industrials", "Industrial Conglomerates", None),
    "SMT.L": (
        "Scottish Mortgage Investment Trust PLC",
        "Financials",
        "Asset Management & Custody Banks",
        None,
    ),
    "FDM.L": (
        "FDM Group (Holdings) plc",
        "Information Technology",
        "IT Consulting & Other Services",
        None,
    ),
    "PSN.L": ("Persimmon Plc", "Consumer Discretionary", "Homebuilding", None),
    "TW.L": ("Taylor Wimpey plc", "Consumer Discretionary", "Homebuilding", None),
    "CRDA.L": ("Croda International Plc", "Materials", "Specialty Chemicals", None),
    "JMAT.L": ("Johnson Matthey Plc", "Materials", "Specialty Chemicals", None),
    "HLMA.L": ("Halma plc", "Information Technology", "Electronic Equipment & Instruments", None),
    "SPX.L": (
        "Spirax Group plc",
        "Industrials",
        "Industrial Machinery & Supplies & Components",
        None,
    ),
    "WEIR.L": (
        "The Weir Group PLC",
        "Industrials",
        "Industrial Machinery & Supplies & Components",
        None,
    ),
    "IMI.L": ("IMI plc", "Industrials", "Industrial Machinery & Supplies & Components", None),
    "SN.L": ("Smith & Nephew plc", "Health Care", "Health Care Equipment", None),
    "SGE.L": ("The Sage Group plc", "Information Technology", "Application Software", None),
    "RS1.L": ("RS Group plc", "Industrials", "Trading Companies & Distributors", None),
    "BEZ.L": ("Beazley plc", "Financials", "Property & Casualty Insurance", None),
    "HIK.L": ("Hikma Pharmaceuticals PLC", "Health Care", "Pharmaceuticals", None),
    "IHG.L": (
        "InterContinental Hotels Group plc",
        "Consumer Discretionary",
        "Hotels, Resorts & Cruise Lines",
        None,
    ),
    "LGEN.L": ("Legal & General Group Plc", "Financials", "Life & Health Insurance", None),
    "STJ.L": ("St. James's Place plc", "Financials", "Asset Management & Custody Banks", None),
    "UTG.L": ("The Unite Group plc", "Real Estate", "Residential REITs", None),
    "ITRK.L": ("Intertek Group plc", "Industrials", "Research & Consulting Services", None),
    "BKG.L": ("The Berkeley Group Holdings plc", "Consumer Discretionary", "Homebuilding", None),
    "SGRO.L": ("SEGRO plc", "Real Estate", "Industrial REITs", None),
    "PSON.L": ("Pearson plc", "Consumer Discretionary", "Education Services", None),
    "DCC.L": ("DCC plc", "Industrials", "Industrial Conglomerates", None),
    "SBRY.L": ("J Sainsbury plc", "Consumer Staples", "Food Retail", None),
    "FRAS.L": ("Frasers Group plc", "Consumer Discretionary", "Apparel Retail", None),
    "RMV.L": ("Rightmove plc", "Communication Services", "Interactive Media & Services", None),
    "WISE.L": ("Wise plc", "Financials", "Transaction & Payment Processing Services", None),
    "EMG.L": ("Man Group plc", "Financials", "Asset Management & Custody Banks", None),
    "MONY.L": ("MONY Group plc", "Communication Services", "Interactive Media & Services", None),
    "EDV.L": ("Endeavour Mining plc", "Materials", "Gold", None),
    "GAW.L": ("Games Workshop Group PLC", "Consumer Discretionary", "Leisure Products", None),
    "IGG.L": ("IG Group Holdings plc", "Financials", "Investment Banking & Brokerage", None),
    "VTY.L": ("Vistry Group PLC", "Consumer Discretionary", "Homebuilding", None),
    "WIZZ.L": ("Wizz Air Holdings Plc", "Industrials", "Passenger Airlines", None),
    "GNS.L": ("Genus plc", "Health Care", "Biotechnology", None),
    "BOY.L": ("Bodycote plc", "Industrials", "Industrial Machinery & Supplies & Components", None),
    "ITV.L": ("ITV plc", "Communication Services", "Broadcasting", None),
    "GFTU.L": ("Grafton Group plc", "Industrials", "Trading Companies & Distributors", None),
    "TRN.L": ("Trainline Plc", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", None),
    "EZJ.L": ("easyJet plc", "Industrials", "Passenger Airlines", None),
}

EXCHANGE_MIC_BY_NAME: dict[str, str] = {
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "LSE": "XLON",
}


def build_universe_metadata() -> dict[str, Any]:
    """Build and validate the complete 286-ticker catalog payload."""
    manifest_path = REPO_ROOT / "data" / "tri_exchange" / "manifest_broad_300.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_tickers = tuple(manifest["valid_tickers"])

    ordered = NASDAQ_TICKERS + NYSE_TICKERS + LSE_TICKERS
    if ordered != manifest_tickers:
        raise ValueError("Exchange ticker lists do not match manifest_broad_300.json order")
    if len(NASDAQ_TICKERS) != 98 or len(NYSE_TICKERS) != 97 or len(LSE_TICKERS) != 91:
        raise ValueError("Unexpected exchange counts (expected 98 NASDAQ / 97 NYSE / 91 LSE)")
    if set(CURATED_CATALOG.keys()) != set(ordered):
        missing = set(ordered) - set(CURATED_CATALOG.keys())
        extra = set(CURATED_CATALOG.keys()) - set(ordered)
        raise ValueError(f"Curated catalog mismatch: missing={missing}, extra={extra}")

    nasdaq_set = set(NASDAQ_TICKERS)
    nyse_set = set(NYSE_TICKERS)

    tickers_payload: dict[str, dict[str, Any]] = {}
    for sym in ordered:
        name, raw_sector, industry, cik = CURATED_CATALOG[sym]
        sector = SECTOR_CROSSWALK.get(raw_sector, raw_sector)
        if sector not in GICS_SECTORS:
            raise ValueError(f"Invalid GICS sector {sector!r} for {sym}")
        if not name or name.upper() == sym.upper():
            raise ValueError(f"Missing or placeholder company name for {sym}")
        if not industry:
            raise ValueError(f"Missing industry for {sym}")

        if sym in nasdaq_set:
            exchange_name = "NASDAQ"
        elif sym in nyse_set:
            exchange_name = "NYSE"
        else:
            exchange_name = "LSE"
        exchange_mic = EXCHANGE_MIC_BY_NAME[exchange_name]

        override = LSE_QUOTE_UNIT_OVERRIDES.get(sym)
        if override is not None:
            currency = override["currency"]
            currency_symbol = override["currency_symbol"]
        elif exchange_name == "LSE":
            currency = "GBp"
            currency_symbol = "p"
        else:
            currency = "USD"
            currency_symbol = "$"

        if sym in CIK_OVERRIDES:
            cik = CIK_OVERRIDES[sym]

        if exchange_name == "LSE":
            if cik is not None:
                raise ValueError(f"LSE ticker {sym} must have cik=None")
        else:
            if cik is None or not re.fullmatch(r"\d{10}", cik):
                raise ValueError(f"US ticker {sym} has invalid 10-digit CIK: {cik!r}")

        tickers_payload[sym] = {
            "ticker": sym,
            "name": name,
            "exchange": exchange_name,
            "exchange_name": exchange_name,
            "exchange_mic": exchange_mic,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "sector": sector,
            "industry": industry,
            "cik": cik,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_date": GENERATED_DATE,
        "total_count": len(ordered),
        "us_count": len(NASDAQ_TICKERS) + len(NYSE_TICKERS),
        "lse_count": len(LSE_TICKERS),
        "exchange_counts": {
            "NASDAQ": len(NASDAQ_TICKERS),
            "NYSE": len(NYSE_TICKERS),
            "LSE": len(LSE_TICKERS),
        },
        "gics_sectors": list(GICS_SECTORS),
        "sector_crosswalk": dict(SECTOR_CROSSWALK),
        "currency_overrides": {
            sym: {
                "currency": info["currency"],
                "currency_symbol": info["currency_symbol"],
            }
            for sym, info in sorted(LSE_QUOTE_UNIT_OVERRIDES.items())
        },
        "cik_overrides": dict(CIK_OVERRIDES),
        "valid_tickers": list(ordered),
        "tickers": tickers_payload,
    }


def _format_js_ticker_array(name: str, tickers: tuple[str, ...], per_line: int = 10) -> str:
    lines = [f"export const {name} = ["]
    for i in range(0, len(tickers), per_line):
        chunk = ", ".join(f"'{sym}'" for sym in tickers[i : i + per_line])
        lines.append(f"  {chunk},")
    lines.append("];")
    return "\n".join(lines)


def render_frontend_universe_js(catalog: dict[str, Any]) -> str:
    """Render deterministic ``frontend/src/universe.js`` from the validated catalog."""
    lines = [
        "/**",
        " * Flagship Tri-Exchange Universal Ticker Definitions & Metadata Catalog (286 Symbols)",
        " * Generated by scripts/generate_universe_metadata.py — DO NOT EDIT BY HAND.",
        " * - NASDAQ: 98 validated constituents (XNAS, USD)",
        " * - NYSE: 97 validated blue-chips & leaders (XNYS, USD)",
        " * - LSE: 91 validated FTSE-100 constituents (XLON, GBp + USD overrides for CPG.L, IHG.L)",
        " */",
        "",
        _format_js_ticker_array("NASDAQ_TICKERS", NASDAQ_TICKERS),
        "",
        _format_js_ticker_array("NYSE_TICKERS", NYSE_TICKERS),
        "",
        _format_js_ticker_array("LSE_TICKERS", LSE_TICKERS),
        "",
        "export const ALL_VALID_TICKERS = [...NASDAQ_TICKERS, ...NYSE_TICKERS, ...LSE_TICKERS];",
        "export const ALL_TICKERS_SET = new Set(ALL_VALID_TICKERS);",
        "export const SUPPORTED_TICKERS = ALL_VALID_TICKERS;",
        "export const SUPPORTED_TICKER_SET = ALL_TICKERS_SET;",
        "",
        "export const TICKER_METADATA = Object.freeze({",
    ]

    tickers_map: dict[str, dict[str, Any]] = catalog["tickers"]
    for sym in catalog["valid_tickers"]:
        entry = tickers_map[sym]
        frontend_entry = {
            "name": entry["name"],
            "exchange": entry["exchange_name"],
            "exchange_name": entry["exchange_name"],
            "exchange_mic": entry["exchange_mic"],
            "currency": entry["currency"],
            "currency_symbol": entry["currency_symbol"],
            "sector": entry["sector"],
        }
        lines.append(f"  {json.dumps(sym)}: {json.dumps(frontend_entry, ensure_ascii=False)},")

    lines.extend(
        [
            "});",
            "",
            "export function normalizeTicker(value) {",
            "  return typeof value === 'string' ? value.trim().toUpperCase() : '';",
            "}",
            "",
            "export function isSupportedTicker(value) {",
            "  return SUPPORTED_TICKER_SET.has(normalizeTicker(value));",
            "}",
            "",
            "export function getTickerMetadata(value) {",
            "  const symbol = normalizeTicker(value);",
            "  return TICKER_METADATA[symbol] || null;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-out",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "universe_metadata.json",
        help="Output path for backend/data/universe_metadata.json",
    )
    parser.add_argument(
        "--frontend-out",
        type=Path,
        default=REPO_ROOT / "frontend" / "src" / "universe.js",
        help="Output path for frontend/src/universe.js",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated files on disk match without writing",
    )
    args = parser.parse_args()

    catalog = build_universe_metadata()
    json_text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    js_text = render_frontend_universe_js(catalog)

    if args.check:
        if not args.backend_out.is_file() or not args.frontend_out.is_file():
            print("Catalog files missing on disk.", file=sys.stderr)
            return 1
        disk_json = args.backend_out.read_text(encoding="utf-8")
        disk_js = args.frontend_out.read_text(encoding="utf-8")
        if disk_json != json_text or disk_js != js_text:
            print("Catalog drift detected.", file=sys.stderr)
            return 1
        print("Catalog parity verified (286 tickers: 98 NASDAQ / 97 NYSE / 91 LSE).")
        return 0

    args.backend_out.parent.mkdir(parents=True, exist_ok=True)
    args.backend_out.write_text(json_text, encoding="utf-8", newline="\n")
    args.frontend_out.parent.mkdir(parents=True, exist_ok=True)
    args.frontend_out.write_text(js_text, encoding="utf-8", newline="\n")
    print(
        f"Generated {args.backend_out.relative_to(REPO_ROOT)} and "
        f"{args.frontend_out.relative_to(REPO_ROOT)} "
        f"({catalog['total_count']} tickers: {catalog['exchange_counts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
