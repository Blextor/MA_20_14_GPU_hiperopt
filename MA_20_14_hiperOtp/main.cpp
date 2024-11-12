#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <windows.h>
#include <sstream>
#include <algorithm>
#include <set>
#include <thread>
#include <random>
#include <list>

using namespace std;

/// Függvény, amely beolvassa a "stocks.txt" fájlból a részvény neveket egy set-be.
std::unordered_set<std::string> readStockNames(const std::string& filename) {
    std::unordered_set<std::string> stockNames;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Hiba a fájl megnyitásakor: " << filename << std::endl;
        return stockNames;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            for (char& c : line) {
                c = std::tolower(c);
            }
            stockNames.insert(line); // Részvény neveket egy set-be mentjük.
        }
    }

    return stockNames;
}

/// Függvény, amely bejárja a megadott mappát, és kigyûjti a megfelelõ fájlok elérési útvonalát.
void findStockFiles(const std::unordered_set<std::string>& stockNames, const std::string& directory, std::vector<std::string>& foundFiles) {
    std::string searchPath = directory + "\\*.us.txt";
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile(searchPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "Nem található fájl ebben a mappában: " << directory << std::endl;
        return;
    }

    do {
        std::string filename = findFileData.cFileName;

        // Levágjuk a ".us.txt" részt a fájl nevébõl.
        std::string stockName = filename.substr(0, filename.find(".us.txt"));

        // Ellenõrizzük, hogy a részvény neve szerepel-e a listában.
        if (stockNames.find(stockName) != stockNames.end()) {
            foundFiles.push_back(directory + "\\" + filename);
        }

    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);
}

/// Függvény, ami összegyűjti a részvények adatait tároló fájlok elérési útvonalát
vector<string> reszvenyekEleresiUtja(string fname="stocks.txt", string folder="data"){
    std::string stockListFile = fname; // A részvény nevek fájlja
    std::unordered_set<std::string> stockNames = readStockNames(stockListFile);

    std::vector<std::string> foundFiles;

    if (stockNames.empty()) {
        std::cerr << "Nincs elérhetõ részvény név." << std::endl;
        return foundFiles;
    }


    std::vector<std::string> directories = {
        "\\daily\\us\\nasdaq stocks\\1",
        "\\daily\\us\\nasdaq stocks\\2",
        "\\daily\\us\\nasdaq stocks\\3",
        "\\daily\\us\\nyse stocks\\1",
        "\\daily\\us\\nyse stocks\\2"
    };

    // Végigmegyünk az összes mappán és megkeressük a fájlokat.
    for (const auto& dir : directories) {
        stringstream ss; ss<<folder<<dir;
        findStockFiles(stockNames, ss.str(), foundFiles);
    }

    // Kiírjuk az összegyûjtött fájlok elérési útvonalait.
    ///std::cout << "Talált fájlok:\n";
    for (const auto& file : foundFiles) {
        std::cout << file << std::endl;
    }
    return foundFiles;
}

/// Függvény, amely felosztja a szöveget megadott karakterek mentén
std::vector<std::string> split(const std::string& text, const std::string& delimiters) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(text);

    while (std::getline(ss, token)) {
        std::size_t start = 0, end = 0;

        while ((end = text.find_first_of(delimiters, start)) != std::string::npos) {
            if (end != start) {
                tokens.push_back(text.substr(start, end - start));
            }
            start = end + 1;
        }

        if (start < text.size()) {
            tokens.push_back(text.substr(start));
        }
    }

    return tokens;
}

struct Datum{
    int ev, honap, nap;

    Datum(int e, int h, int n){ev=e; honap=h; nap=n;}
    Datum(){ev=0; honap=0; nap=0;}

    int toInt(){return ev*10000+honap*100+nap;}
    Datum fromInt(int d){
        Datum datum; datum.ev=d/10000;
        datum.honap=(d%10000)/100;
        datum.nap=d%100;
        return datum;
    }

    void setInt(int d){ev=d/10000;
        honap=(d%10000)/100;
        nap=d%100;
    }

    string toString(){
        stringstream ss; ss<<ev<<" "<<honap<<" "<<nap;
        return ss.str();
    }

    string toStringDate(){
        stringstream ss; ss<<ev<<"."<<honap<<"."<<nap<<".";
        return ss.str();
    }

    bool operator==(const Datum& other) const {
        return (ev==other.ev && honap==other.honap && nap==other.nap);
    }

    bool operator<(const Datum& other) const {
        if (ev<other.ev) return true;
        if (ev==other.ev && honap<other.honap) return true;
        if (ev==other.ev && honap==other.honap && nap<other.nap) return true;
        return false;
    }
};

struct MozgoAtlag{
    int hossz;
    vector<float> atlag;
    vector<Datum> datum;
    vector<float> zaras;
    vector<float> nyitas;
};

struct Nap{
    float nyitas, minimum, maximum, zaras;
    Datum datum;
};

struct Reszveny{
    string nev;
    vector<Nap> napok;
    vector<MozgoAtlag> mozgoatlag;
};

struct ReszvenyGPU{
    vector< vector<float> > mozgoatlagokAtlag;
    vector<float> mozgoatlagokZaras;
    vector<float> mozgoatlagokDatum;
    int N;
};

/// Betölti a fájlokból a részvények adatait
vector<Reszveny> reszvenyekBetoltese(vector<string> fajlnevek){
    vector<Reszveny> reszvenyek;
    /// most az adatokat kell beolvasni az adott fájlokból
    for (size_t i=0; i<fajlnevek.size(); i++){
        ifstream ifileStockData(fajlnevek[i]);
        Reszveny reszveny;
        string reszvenyNeve,fejlec,toSplit;
        ifileStockData>>fejlec;
        vector<string> elemek;
        reszveny.mozgoatlag.resize(200);
        while (!ifileStockData.eof()){
            ifileStockData>>toSplit;
            elemek = split(toSplit,",");
            stringstream ss(elemek[2]); /// 19871230 -> 1987 12 30
            int datum=0; ss>>datum;
            Nap nap; nap.datum.ev=datum/10000; nap.datum.nap=datum%100;
            nap.datum.honap=(datum%10000)/100;
            nap.nyitas=stof(elemek[4]);
            nap.maximum=stof(elemek[5]);
            nap.minimum=stof(elemek[6]);
            nap.zaras=stof(elemek[7]);
            reszveny.napok.push_back(nap);

            for (float n=1.0f; n<=200; n++){
                if (reszveny.napok.size()<n) continue;
                if (reszveny.napok.size()==n) {
                    /*
                    MozgoAtlag ujAtlag; ujAtlag.datum=nap.datum;
                    for (int k=0; k<200;k++) ujAtlag.m200+=reszveny.napok[k].zaras;
                    ujAtlag.m200/=200.0f;
                    for (int k=100; k<200;k++) ujAtlag.m100+=reszveny.napok[k].zaras;
                    ujAtlag.m100/=100.0f;
                    for (int k=150; k<200;k++) ujAtlag.m50+=reszveny.napok[k].zaras;
                    ujAtlag.m50/=50.0f;
                    ujAtlag.zaras=reszveny.napok[199].zaras;
                    reszveny.mozgoatlag.push_back(ujAtlag);
                    */
                    continue;
                }
                reszveny.mozgoatlag[n-1].hossz=n;
                reszveny.mozgoatlag[n-1].datum.push_back(nap.datum);
                reszveny.mozgoatlag[n-1].zaras.push_back(nap.zaras);
                reszveny.mozgoatlag[n-1].nyitas.push_back(nap.nyitas);
                float atlag = 0;
                for (int k=reszveny.napok.size()-n; k<reszveny.napok.size();k++)
                    atlag+=reszveny.napok[k].zaras;
                atlag/=n;
                reszveny.mozgoatlag[n-1].atlag.push_back(atlag);
                /*
                MozgoAtlag ujAtlag = reszveny.mozgoatlag.back();
                int index = reszveny.napok.size()-201;
                ujAtlag.datum = nap.datum;
                ujAtlag.m200+=nap.zaras/200.0f;
                ujAtlag.m200-=reszveny.napok[index].zaras/200.0f;
                ujAtlag.m100+=nap.zaras/100.0f;
                ujAtlag.m100-=reszveny.napok[index+100].zaras/100.0f;
                ujAtlag.m50+=nap.zaras/50.0f;
                ujAtlag.m50-=reszveny.napok[index+150].zaras/50.0f;
                ujAtlag.zaras=reszveny.napok[index+200].zaras;
                reszveny.mozgoatlag.push_back(ujAtlag);
                */
            }
        }
        elemek = split(elemek[0],".");
        reszveny.nev=elemek[0];
        cout<<reszveny.nev<<endl;
        reszvenyek.push_back(reszveny);
    }
    return reszvenyek;
}

void reszvenyBetoltese(Reszveny& reszveny, string fajlnev){
    reszveny.mozgoatlag.clear(); reszveny.napok.clear(); reszveny.nev="";
    ifstream ifileStockData(fajlnev);
    string reszvenyNeve,fejlec,toSplit;
    ifileStockData>>fejlec;
    vector<string> elemek;
    reszveny.mozgoatlag.resize(200);
    vector<list<MozgoAtlag>> mozgAtlgk(200);
    while (!ifileStockData.eof()){
        ifileStockData>>toSplit;
        elemek = split(toSplit,",");
        stringstream ss(elemek[2]); /// 19871230 -> 1987 12 30
        int datum=0; ss>>datum;
        Nap nap; nap.datum.ev=datum/10000; nap.datum.nap=datum%100;
        nap.datum.honap=(datum%10000)/100;
        nap.nyitas=stof(elemek[4]);
        nap.maximum=stof(elemek[5]);
        nap.minimum=stof(elemek[6]);
        nap.zaras=stof(elemek[7]);
        reszveny.napok.push_back(nap);
        ///if (nap.datum.ev<2000) continue;

        for (float n=1.0f; n<=200; n++){
            /*
            if (reszveny.mozgoatlag[n-1].atlag.size()==0 && reszveny.napok[reszveny.napok.size()-2].datum.ev<2000){
                Datum datum;
                for (int o=0;o<200-n+1; o++){
                    reszveny.mozgoatlag[n-1].atlag.push_back(0.0f);
                    reszveny.mozgoatlag[n-1].datum.push_back(datum);
                    reszveny.mozgoatlag[n-1].zaras.push_back(0.0f);
                    reszveny.mozgoatlag[n-1].nyitas.push_back(0.0f);
                }
            }
            */
            if (reszveny.napok.size()<n) continue;
            if (reszveny.napok.size()==n) {
                /*
                MozgoAtlag ujAtlag; ujAtlag.datum=nap.datum;
                for (int k=0; k<200;k++) ujAtlag.m200+=reszveny.napok[k].zaras;
                ujAtlag.m200/=200.0f;
                for (int k=100; k<200;k++) ujAtlag.m100+=reszveny.napok[k].zaras;
                ujAtlag.m100/=100.0f;
                for (int k=150; k<200;k++) ujAtlag.m50+=reszveny.napok[k].zaras;
                ujAtlag.m50/=50.0f;
                ujAtlag.zaras=reszveny.napok[199].zaras;
                reszveny.mozgoatlag.push_back(ujAtlag);
                */
                continue;
            }
            reszveny.mozgoatlag[n-1].hossz=n;
            reszveny.mozgoatlag[n-1].datum.push_back(nap.datum);
            reszveny.mozgoatlag[n-1].zaras.push_back(nap.zaras);
            reszveny.mozgoatlag[n-1].nyitas.push_back(nap.nyitas);
            float atlag = 0;
            for (int k=reszveny.napok.size()-n; k<reszveny.napok.size();k++)
                atlag+=reszveny.napok[k].zaras;
            atlag/=n;
            reszveny.mozgoatlag[n-1].atlag.push_back(atlag);
            /*
            MozgoAtlag ujAtlag = reszveny.mozgoatlag.back();
            int index = reszveny.napok.size()-201;
            ujAtlag.datum = nap.datum;
            ujAtlag.m200+=nap.zaras/200.0f;
            ujAtlag.m200-=reszveny.napok[index].zaras/200.0f;
            ujAtlag.m100+=nap.zaras/100.0f;
            ujAtlag.m100-=reszveny.napok[index+100].zaras/100.0f;
            ujAtlag.m50+=nap.zaras/50.0f;
            ujAtlag.m50-=reszveny.napok[index+150].zaras/50.0f;
            ujAtlag.zaras=reszveny.napok[index+200].zaras;
            reszveny.mozgoatlag.push_back(ujAtlag);
            */
        }
    }
    elemek = split(elemek[0],".");
    reszveny.nev=elemek[0];
    cout<<reszveny.nev<<endl;
}

vector<Reszveny> reszvenyekParhuzamosBetoltese(vector<string> fajlnevek){
    vector<Reszveny> osszesReszveny; osszesReszveny.reserve(100000);
    int thCnt = 4;
    vector<thread> szalak; szalak.resize(thCnt);
    vector<Reszveny> reszvenyek; reszvenyek.resize(thCnt);

    cout<<"CNT: "<<fajlnevek.size()<<endl;
    clock_t t2 = clock();
    clock_t t3 = clock();
    for (size_t i=0; i<fajlnevek.size();){
        int savedI = i;
        for (int j=0; j<thCnt; j++){
            szalak[j] = thread(reszvenyBetoltese,ref(reszvenyek[j]),fajlnevek[i]);
            i++;
            if (i>=fajlnevek.size()) break;
        }

        float ts = 0;
        for (int j=0; j<thCnt; j++){
            if (savedI+j>=fajlnevek.size()) continue;
            clock_t t1 = clock();
            if (szalak[j].joinable())
                szalak[j].join();
            if (clock()-t2>=5000){
                cout<<savedI+j<<" "<<clock()-t2<<" i "<<ts<<" - "<<(float)(clock()-t3)/(float)(savedI+j)<<endl;
                t2=clock();
            }
            osszesReszveny.push_back(reszvenyek[j]);
        }
    }
    //for (int i=0; i<thCnt; i++)
      //  szalak[i].join();
    return osszesReszveny;
}


char xedikKarakter(int X, string abc = "abcdefghijklmnopqrstuvwxyz"){
    int i = 0;
    for(char& c : abc) {
        i++;
        if (i==X) return c;
    }
    return '?';
}

int karakterSorszama(char X){
    string abc = "abcdefghijklmnopqrstuvwxyz";
    for(int i=0; i<20; i++) {
        if (abc[i]==X) return i;
    }
    return -1;
}

struct Par{
    float ertek;
    char karakter;

    bool operator<(const Par& other) const {
        return ertek<other.ertek;
    }
};

/*
set<float> convertToSet(vector<float> v)
{
    set<float> s;
    for (float x : v) {
        s.insert(x);
    }
    return s;
}
*/

template <typename T>
std::set<T> convertToSet(const std::vector<T>& v) {
    std::set<T> s;
    for (const T& x : v) {
        s.insert(x);
    }
    return s;
}


struct Pelda {
    string stockName;
    Datum datum;
    int reszvenyIdx, mozgoatlagIdx;

    bool operator<(const Pelda& other) const {
        if (datum==other.datum)
            return stockName<other.stockName;
        return datum<other.datum;
    }
};

struct PeldaGPU{
    int reszvenyIdx, mozgoatlagIdx, datum;

    bool operator<(const PeldaGPU& other) const {
        if (mozgoatlagIdx==other.mozgoatlagIdx)
            return reszvenyIdx<other.reszvenyIdx;
        return mozgoatlagIdx<other.mozgoatlagIdx;
    }
};

struct Eset{
    string charChain= "";
    float osszesEset = 0;
    float pozitivEset = 0;

    float szum = 0, prod = 1;

    vector<Pelda> peldak;

    bool operator<(const Eset& other) const {
        return charChain<other.charChain;
    }
    bool operator==(const Eset& other) const {
        return charChain==other.charChain;
    }

    // Kényelmes kiíráshoz egy barát operátor
    friend std::ostream& operator<<(std::ostream& os, const Eset& obj) {
        //return os << obj.charChain << ": " << obj.osszesEset << ", " << obj.pozitivEset << ", " << obj.szum << ", " << obj.prod;

        return os << obj.charChain << " "
        << obj.osszesEset<< " " << obj.pozitivEset << " "
        << obj.szum << " " << obj.prod << " "
        << obj.pozitivEset/obj.osszesEset*100.0f << " "
        << obj.szum/obj.osszesEset<<" ";
    }
};

struct Par2{
    float ertek;
    float sorszam;

    bool operator<(const Par2& other) const {
        return ertek<other.ertek;
    }
};

struct Ertek {
    int osszes=0, pozitiv=0;
    float szum=0, prod=1;
};

Datum xhonapra(Datum kezdo, int honapokSZama){
    for (int i=0; i<honapokSZama; i++){
        kezdo.honap++;
        if(kezdo.honap>12) {kezdo.ev++; kezdo.honap=1;}
    }
    return kezdo;
}

struct Tranzakcio{
    Datum datum;
    string reszveny;
    float ertek;

    bool operator<(const Tranzakcio& other) const {
        return datum<other.datum;
    }
};

struct Score{
    vector<float> evVegiek, teljes;
    vector<int> alkalmakEvente;
    vector<Pelda> osszesPelda;
    int egyNapMaximum=0;
    float atlagosProfit=0;
    float atlagosNapiProfit=0;

    void clrt(){
        evVegiek.clear(); evVegiek.resize(25);
        teljes.clear(); osszesPelda.clear();
        alkalmakEvente.clear(); alkalmakEvente.resize(25);
        egyNapMaximum=0;
        atlagosNapiProfit=0;
        atlagosProfit=0;
    }
};

struct Parameterek{
    int adasVeteliNapok = 0; /// (0-5)
    int m1 = 1, m2=2, m3=3; /// a három vizsgált mozgóátlag (1-50, különözőek)
    int ms = 0; /// mozgóátlagok növekvő sorrendje (0-5)
    int mi = 0; /// m1 és m3 növekszik / csökken
    bool buy = 0; /// adni vagy venni kell
    bool toresAlatt = 0; /// kisebb vaagy nagyobb legyen a törésnél?
    float tores = 0.0039f;
};

bool chkSorrend(const float& a, const float& b, const float& c){
    return (a<=b && b<=c);
}

int getScore(vector<Reszveny>& reszvenyek, Parameterek& params, Score& score, vector<Datum>& osszesDatum, int ert){
    score.clrt();
    clock_t stime = clock();
    int MOZGO_ATLAG_MIN_MERET = 50;


    Datum kezdo(2014,8,1); /// 2009, 8
    vector<vector<int>> tores_cnt_per_year; tores_cnt_per_year.resize(25,vector<int> (200,0));
    vector<vector<float>> tores_szum_per_year_pre; tores_szum_per_year_pre.resize(25,vector<float> (200,0));
    vector<vector<vector<float>>> tores_szum_per_year;
    for (int i=0; i<20; i++) tores_szum_per_year.push_back(tores_szum_per_year_pre);
    //for (int i=0; i<)

    bool debug = false;

    vector<Datum> osszesEset; osszesEset.reserve(100000);
    vector<Pelda> osszesPelda; osszesPelda.reserve(100000);
    ///cout<<"ALMA "<<ert<<endl;
    ///return 0;
    Pelda pelda;
    int itrCnt = 0, itrCnt2 = 0, itrCnt3 = 0;
    for (int i=0;i<reszvenyek.size(); i++){
        ///cout<<"R"<<i<<endl;
        for (int k=2; k<reszvenyek[i].mozgoatlag[params.m2].atlag.size(); k++){
            itrCnt++;
            /// megvizsgálja hogy az adott nap megfelelő-e
            int ev = reszvenyek[i].mozgoatlag[params.m2].datum[k].ev - 2000;
            if (ev<0) continue;
            if (k+7>=reszvenyek[i].mozgoatlag[params.m2].atlag.size()) continue;

            itrCnt2++;
            /// ???
            /** Nem kell most
            /// dupla k-1 es eredmény érdekes volt
            if (reszvenyek[i].mozgoatlag[19].atlag[k-1]<reszvenyek[i].mozgoatlag[19].atlag[k] ==
                reszvenyek[i].mozgoatlag[13].atlag[k-1+6]<reszvenyek[i].mozgoatlag[13].atlag[k+6])
            {
                torolt=true;
                continue;
            }
            */

            /// törés2

            float a1 = reszvenyek[i].mozgoatlag[params.m2].atlag[k-2];
            float a2 = reszvenyek[i].mozgoatlag[params.m2].atlag[k-1];
            float a3 = reszvenyek[i].mozgoatlag[params.m2].atlag[k-0];
            float b1 = a2-a1, b2 = a3-a2;

            int im1 = params.m2-params.m1;
            int im3 = params.m2-params.m3;

            bool V1 = false, V2 = true;
            /// V1
            if (V1){
                if (!params.toresAlatt){
                    if (b2<=0 || b1<=0 || b1<b2) continue;
                    if (b1-b2<=a2*params.tores) continue;
                } else {
                    if (b2>0 || b1>0 || b1>b2) continue;
                    if (b1-b2<a2*params.tores) continue;
                }
            }

            /// V2
            if (V2){
                if ((b2<=0 || b1<=0 || b1<b2) && (b2>0 || b1>0 || b1>b2)) continue;
                if (!params.toresAlatt){
                    if (b1-b2<=a2*params.tores) continue;
                } else {
                    if (b1-b2>=a2*params.tores) continue;
                }
            }

            if (k+im3-1<0) continue;

            /// mozgó átlagok közötti növekvő sorrend check
            float ma1=reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1];
            float ma2=reszvenyek[i].mozgoatlag[params.m2].atlag[k+0];
            float ma3=reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3];

            if (params.ms==0) {if (!chkSorrend(ma3,ma2,ma1)) continue;}
            else if (params.ms==1) {if (!chkSorrend(ma3,ma1,ma2)) continue;}
            else if (params.ms==2) {if (!chkSorrend(ma2,ma1,ma3)) continue;}
            else if (params.ms==3) {if (!chkSorrend(ma2,ma3,ma1)) continue;}
            else if (params.ms==4) {if (!chkSorrend(ma1,ma3,ma2)) continue;}
            else if (params.ms==5) {if (!chkSorrend(ma1,ma2,ma3)) continue;}

            /// mozgóátlagok maguk növekvő/csökkenő
            if (params.mi==0){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] <0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] <0) continue;
            }
            else if (params.mi==1){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] >=0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] <0) continue;
            }
            else if (params.mi==2){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] <0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] >=0) continue;
            }
            else if (params.mi==3){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] >=0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] >=0) continue;
            }


            itrCnt3++;


            ///cout<<score.alkalmakEvente.size()<<" "<<tores_cnt_per_year[ev][params.m2]<<endl;
            tores_cnt_per_year[ev][params.m2]++;
            score.alkalmakEvente[ev]++;
            ///continue;
            pelda.stockName=reszvenyek[i].nev; pelda.datum=reszvenyek[i].mozgoatlag[params.m2].datum[k];
            pelda.reszvenyIdx=i; pelda.mozgoatlagIdx=k;
            osszesPelda.push_back(pelda);
            osszesEset.push_back(reszvenyek[i].mozgoatlag[params.m2].datum[k]);
            //xnapja=0;

        }
    }
    ///cout<<"ITR: "<<itrCnt<<" "<<itrCnt2<<" "<<itrCnt3<<endl;

    if (itrCnt3>50000) score;
    clock_t midtime = clock();
    ///cout<<"BALMA "<<ert<<endl;
    ///return 0;

    ///set<Datum> datumok = convertToSet(osszesEset);
    sort(osszesPelda.begin(),osszesPelda.end());
    list<Pelda> osszesPeldaList;
    osszesPeldaList.insert(osszesPeldaList.end(),osszesPelda.begin(),osszesPelda.end());

    vector<float> napiErtek;
    napiErtek.resize(osszesDatum.size(),100.0f);

    int ev = 2000;
    float tempSzum = 0, tempCnt = 0;
    vector<float> eviErtekek;
    vector<vector<float>> eviNapok;
    ///cout<<"B2ALMA "<<ert<<endl;
    clock_t midtime2 = clock();
///    if (ert==-1) score.osszesPelda=osszesPelda;

    float peldasNapCnt=0;
    for (int i=0; i<osszesDatum.size(); i++){
        ///if (i%100==0)cout<<"O"<<i<<endl;
        ///continue;
        if (i+1<napiErtek.size())
            napiErtek[i+1]=napiErtek[i];
        if (osszesDatum[i].ev>ev){
            ev=osszesDatum[i].ev;
            ///napiErtek[i+0]=100;
            ///napiErtek[i+1]=100;
            score.evVegiek[ev-2001]=eviErtekek.back();
            eviNapok.push_back(eviErtekek);
            eviErtekek.clear();
        }
        vector<float> ertekek;
        ///if (datumok.find(osszesDatum[i])==datumok.end()){eviErtekek.push_back(napiErtek[i]); continue;}
        ///for (int j=0; j<osszesPelda.size(); j++){
        bool vanPelda = false;
        while(osszesPeldaList.front().datum==osszesDatum[i]){
            vanPelda=true;
            ///if (osszesPelda[j].datum==osszesDatum[i]){
                int stockIdx = osszesPeldaList.front().reszvenyIdx;
                int mozgoAtlagIdx = osszesPeldaList.front().mozgoatlagIdx;
                if (mozgoAtlagIdx+3>= reszvenyek[stockIdx].mozgoatlag[params.m2].zaras.size()) cout<<"ZZZZZZZ"<<endl;
                bool felfele = params.buy;///reszvenyek[stockIdx].mozgoatlag[params.m2].atlag[mozgoAtlagIdx]>reszvenyek[stockIdx].mozgoatlag[params.m2].atlag[mozgoAtlagIdx-1];
                int aznapiVetel = params.adasVeteliNapok; /// MELYIK METÓDUS
                if (aznapiVetel==0){
                    float aznapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx];
                    float masnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+1];
                    if (felfele) ertekek.push_back(masnapiZaras/aznapiZaras-1.0f);
                    else ertekek.push_back(-(masnapiZaras/aznapiZaras-1.0f));
                } else if (aznapiVetel==1){
                    float masnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+1];
                    float harmadnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+2];
                    if (felfele) ertekek.push_back(harmadnapiZaras/masnapiZaras-1.0f);
                    else ertekek.push_back(-(harmadnapiZaras/masnapiZaras-1.0f));
                } else if (aznapiVetel==2) {
                    float harmadnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+2];
                    float negyednapZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+3];
                    if (felfele) ertekek.push_back(negyednapZaras/harmadnapiZaras-1.0f);
                    else ertekek.push_back(-(negyednapZaras/harmadnapiZaras-1.0f));
                } else if (aznapiVetel==3){
                    float aznapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx];
                    float harmadnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+2];
                    if (felfele) ertekek.push_back(harmadnapiZaras/aznapiZaras-1.0f);
                    else ertekek.push_back(-(harmadnapiZaras/aznapiZaras-1.0f));
                } else if (aznapiVetel==4) {
                    float masnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+1];
                    float negyednapZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+3];
                    if (felfele) ertekek.push_back(negyednapZaras/masnapiZaras-1.0f);
                    else ertekek.push_back(-(negyednapZaras/masnapiZaras-1.0f));
                } else if (aznapiVetel==5) {
                    float aznapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx];
                    float negyednapZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+3];
                    if (felfele) ertekek.push_back(negyednapZaras/aznapiZaras-1.0f);
                    else ertekek.push_back(-(negyednapZaras/aznapiZaras-1.0f));
                }
            ///}
            osszesPeldaList.pop_front();
        }
        if (vanPelda) peldasNapCnt++;
        int oszto = ertekek.size();
        float foszto = oszto;
        score.egyNapMaximum=max(oszto,score.egyNapMaximum);
        ///cout<<score.egyNapMaximum<<endl;

        for (int j=0; j<oszto; j++){
            float tempSum = 0;
            if (i+1<napiErtek.size()){
                napiErtek[i+1]+=napiErtek[i]*ertekek[j]/foszto;
                tempSzum+=ertekek[j];
                tempSum+=ertekek[j];;
                tempCnt++;
            }
            score.atlagosNapiProfit+=tempSum/foszto;
        }
        eviErtekek.push_back(napiErtek[i]);
    }
    score.atlagosNapiProfit/=max(1.0f,peldasNapCnt);
    clock_t endtime=clock();
    score.evVegiek[24]=eviErtekek.back();
    eviNapok.push_back(eviErtekek);
    ///cout<<"CALMA "<<ert<<endl;
    score.atlagosProfit=tempSzum/max(1.0f,tempCnt);
    score.teljes=napiErtek;
    if (ert==-2)cout<<"TIME: "<<endtime-midtime2<<" "<<midtime2-midtime<<" "<<midtime-stime<<endl;
    return 0;
}

vector<Datum> getOsszesDatum(vector<Reszveny>& reszvenyek){
    vector<Datum> osszesDatum;
    int legtobbNap = 0, idx=0;
    for (int i=0;i<reszvenyek.size(); i++){
        if (reszvenyek[i].napok.size()>legtobbNap) {
            legtobbNap=reszvenyek[i].napok.size();
            idx=i;
        }
    }
    for (int i=0;i<reszvenyek[idx].napok.size(); i++){
        if (reszvenyek[idx].napok[i].datum.ev<2000) continue;
        osszesDatum.push_back(reszvenyek[idx].napok[i].datum);
    }

    return osszesDatum;
}

void runTest(vector<Reszveny>& reszvenyek, Parameterek& params, Score& score, vector<Datum>& osszesDatum, int ert){

}

vector<ReszvenyGPU> getReszvenyekGPU(vector<Reszveny>& reszvenyek,vector<Datum>& osszesDatum){
    vector<ReszvenyGPU> ret;
    //vector<float> atlag(osszesDatum.size(),0);
    const int MOZGO_CNT = 50;
    for (int i=0; i<reszvenyek.size(); i++){
        ///cout<<"I "<<i<<" "<<reszvenyek[i].nev<<endl;
        ReszvenyGPU rgpu;
        rgpu.mozgoatlagokAtlag.resize(MOZGO_CNT,vector<float>(osszesDatum.size()));
        rgpu.mozgoatlagokZaras.resize(osszesDatum.size());
        rgpu.mozgoatlagokDatum.resize(osszesDatum.size());
        rgpu.N=osszesDatum.size();
        for (int j=0; j<MOZGO_CNT; j++){
            ///cout<<"J "<<j<<endl;
            int midx = 0;
            //rgpu.mozgoatlagokAtlag[j].resize(osszesDatum.size());
            //rgpu.mozgoatlagokDatum[j].resize(osszesDatum.size());
            for (int z=0; z<osszesDatum.size(); z++){
                ///if (i==469 && j==20 && z==137) cout<<"PEKING"<<endl;
                //cout<<"Z "<<z<<endl;
                ///if (i==2 && j==25 && z==1425) cout<<"ALMA "<<reszvenyek[i].nev<<endl;
                if (reszvenyek[i].mozgoatlag[j].datum[midx]<osszesDatum[z]){
                        ///if (i==469 && j==20 && z==137) cout<<"PEKING2"<<endl;
                    z--;
                    midx++;
                    continue;
                }
                if (osszesDatum[z]<reszvenyek[i].mozgoatlag[j].datum[midx]){
                    ///if (i==469 && j==20 && z==137) cout<<"PEKING1"<<endl;
                    rgpu.mozgoatlagokAtlag[j][z]=-1;
                    rgpu.mozgoatlagokZaras[z]=-1;
                    rgpu.mozgoatlagokDatum[z]=-1;
                    continue;
                }
                ///if (i==2 && j==25 && z==1425) cout<<"ALMA2"<<endl;
                ///cout<<"ok2"<<endl;
                ///if (i==469 && j==20 && z==137) cout<<"PEKING3"<<endl;
                rgpu.mozgoatlagokAtlag[j][z]=reszvenyek[i].mozgoatlag[j].atlag[midx];
                rgpu.mozgoatlagokZaras[z]=reszvenyek[i].mozgoatlag[j].zaras[midx];
                rgpu.mozgoatlagokDatum[z]=reszvenyek[i].mozgoatlag[j].datum[midx].toInt();
                midx++;
            }
        }
        ret.push_back(rgpu);
    }

    return ret;
}

int main(){
    clock_t fullT = clock();
    vector<string> reszvenyekFajlNeve = reszvenyekEleresiUtja("ossz24_09.txt","data");
    vector<Reszveny> reszvenyek = reszvenyekParhuzamosBetoltese(reszvenyekFajlNeve);
    vector<Datum> osszesDatum = getOsszesDatum(reszvenyek);

    bool speedTest = true;
    if (speedTest){
        Parameterek params; params.m1=14; params.m2=20; params.m3=50;
        params.adasVeteliNapok=1;
        params.tores=0.0039f;
        params.ms=0;
        params.mi=0;
        params.buy=true;
        params.toresAlatt=false;
        //parameterek.push_back(params);

        Score score;

        clock_t time1 = clock();
        getScore(reszvenyek,params,score,osszesDatum,-2);
        getScore(reszvenyek,params,score,osszesDatum,-2);
        getScore(reszvenyek,params,score,osszesDatum,-2);
        getScore(reszvenyek,params,score,osszesDatum,-2);
        cout<<clock()-time1<<" "<<clock()-fullT<<endl;
        cout<<score.evVegiek[24]<<endl;
        exit(1);
        return 0;
    }

    clock_t t = clock();
    int cnt = 0;
    vector<Parameterek> parameterek;
    for (int i1=4; i1<=21; i1++){
        for (int i2=i1+5; i2<49; i2++){
            for (int i3=i2+5; i3<=49; i3+=2){

                ///continue;
                for (int j=0; j<2; j++){
                    for (int k=0; k<6;k++){
                        for (int l=0; l<4;l++){
                            for (int b=0;b<2; b++){
                                for (int u=0; u<2; u++){
                                    for (float f=-0.0077f; f<0.0077f; f+=0.0002f){
                                        Parameterek params; params.m1=i1; params.m2=i2; params.m3=i3;
                                        params.adasVeteliNapok=j;
                                        params.tores=f;
                                        params.ms=k;
                                        params.mi=l;
                                        params.toresAlatt=(u==0);
                                        params.buy=(b==0);
                                        parameterek.push_back(params);
                                        cnt++;
                                        ///getScore(reszvenyek,params);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }




    bool test = false;
    if (test){
        ifstream testf("test.txt");
        ofstream oftest("oftest.txt");
        set<Pelda> tesztPeldak;
        int vizsgaltAdasvetel = 0;

        int maxEgynap = 0;

        while(!testf.eof()){
            int idx;
            testf>>idx;
            if (parameterek[idx].adasVeteliNapok!=vizsgaltAdasvetel) continue;
            Score score; getScore(reszvenyek,parameterek[idx],score,osszesDatum,-1);
            if (score.evVegiek[0]<0.90f) continue;
            //if (sco)
            oftest<<score.atlagosNapiProfit<<" "<<score.atlagosProfit<<" "<<score.egyNapMaximum<<" ";//<<endl;
            for (int i=0;i<25;i++){
                oftest<<score.alkalmakEvente[i]<<" ";
            }
            float minmax1 = 10;
            for (int i=0;i<25;i++){
                oftest<<score.evVegiek[i]<<" ";
                if (i==0){
                    minmax1=score.evVegiek[i]/100.0f;
                } else if (i<24)
                    minmax1= min(score.evVegiek[i]/score.evVegiek[i-1],minmax1);
                else {
                    minmax1= min(score.teljes[6225]/score.evVegiek[i-1],minmax1);
                }
            }
            oftest<<endl;
            if (minmax1<0.98f) continue;
            cout<<minmax1<<endl;
            tesztPeldak.insert(score.osszesPelda.begin(),score.osszesPelda.end());
        }
        oftest.close();

        ofstream oftestp("oftestPeldak.txt");
        for (Pelda pelda: tesztPeldak){
            cout<<pelda.datum.toString()<<endl;
            oftestp<<pelda.datum.toString()<<" "<<pelda.stockName<<endl;
        }

        vector<float> napiErtek; napiErtek.resize(osszesDatum.size(),100.0f);
        for (int i=0; i<osszesDatum.size(); i++){

        }
        int ev = osszesDatum[0].ev;
        for (int i=0; i<osszesDatum.size(); i++){
            ///continue;
            if (i+1<napiErtek.size())
                napiErtek[i+1]=napiErtek[i];
            if (osszesDatum[i].ev>ev){
                ev=osszesDatum[i].ev;
                ///napiErtek[i+0]=100;
                ///napiErtek[i+1]=100;
            }
            vector<float> ertekek;
            //if (datumok.find(osszesDatum[i])==datumok.end()){ continue;}
            //for (int j=0; j<osszesPelda.size(); j++){
            for (Pelda pelda: tesztPeldak){
                if (osszesDatum[i]<pelda.datum) continue;
                if (pelda.datum==osszesDatum[i]){
                    int stockIdx = pelda.reszvenyIdx;
                    int mozgoAtlagIdx = pelda.mozgoatlagIdx;
                    int idx = -1;
                    for (int j=0; j<reszvenyek[stockIdx].napok.size(); j++){
                        if (reszvenyek[stockIdx].napok[j].datum==pelda.datum){
                            idx=j;
                            break;
                        }
                    }
                    //bool felfele = reszvenyek[stockIdx].napok[idx].zaras>reszvenyek[stockIdx].mozgoatlag[params.m2].atlag[mozgoAtlagIdx-1];
                    bool felfele = true;
                    int aznapiVetel = vizsgaltAdasvetel; /// MELYIK METÓDUS
                    if (aznapiVetel==0){
                        float aznapiZaras = reszvenyek[stockIdx].napok[idx].zaras;
                        float masnapiZaras = reszvenyek[stockIdx].napok[idx+1].zaras;
                        if (felfele) ertekek.push_back(masnapiZaras/aznapiZaras-1.0f);
                        else ertekek.push_back(-(masnapiZaras/aznapiZaras-1.0f));
                    } else if (aznapiVetel==1){
                        float masnapiZaras = reszvenyek[stockIdx].napok[idx+1].zaras;
                        float harmadnapiZaras = reszvenyek[stockIdx].napok[idx+2].zaras;
                        if (felfele) ertekek.push_back(harmadnapiZaras/masnapiZaras-1.0f);
                        else ertekek.push_back(-(harmadnapiZaras/masnapiZaras-1.0f));
                    }
                }
            }
            int oszto = ertekek.size();
            float foszto = oszto;

            if (oszto>maxEgynap) maxEgynap=oszto;

            for (int j=0; j<oszto; j++){
                if (i+1<napiErtek.size()){
                    napiErtek[i+1]+=napiErtek[i]*ertekek[j]/foszto;
                }
            }
        }

        ofstream o2file("napiAlakulas0_98_24-11.txt");
        for (int i=0; i<osszesDatum.size(); i++){
            o2file<<osszesDatum[i].toStringDate()<<" "<<napiErtek[i]<<endl;
        }
        o2file.close();

        cout<<maxEgynap<<endl;

        return 0;
    }

    int thCnt = 32;
    vector<thread> szalak; szalak.resize(thCnt);
    vector<Score> scores; scores.resize(thCnt);

    vector<Score> scoresAll;

    list<Score> ertekek; ertekek.resize(thCnt);
    cout<<"CNT: "<<cnt<<" "<<parameterek.size()<<endl;
    clock_t t2 = clock();
    clock_t t3 = clock();
    ofstream ofile("nalassuk.txt");
    for (size_t i=28552601; i<parameterek.size();){
        int savedI = i;
        for (int j=0; j<thCnt; j++){
            szalak[j] = thread(getScore,ref(reszvenyek),ref(parameterek[i]),ref(scores[j]),ref(osszesDatum),j);
            i++;

            //cout<<i<<endl;
            if (i>=parameterek.size()) break;
            ///cout<<i<<endl;
        }

        float ts = 0;
        for (int j=0; j<thCnt; j++){
            if (savedI+j>=parameterek.size()) continue;
            clock_t t1 = clock();
            if (szalak[j].joinable())
                szalak[j].join();
            //if ((savedI+j)%1000==0)
            if (clock()-t2>=5000){
                cout<<savedI+j<<" "<<clock()-t2<<" i "<<ts<<" - "<<(float)(clock()-t3)/(float)(savedI+j-28552601)<<endl;
                t2=clock();
            }
            //t2=clock();
            ///cout<<savedI+j<<endl;
            ///cout<<scores[j].diffDays<<endl;
            //scoresAll.push_back(scores[j]);
            //ts+=scores[j].atlagosProfit;
            ofile<<parameterek[savedI+j].m1<<" "<<parameterek[savedI+j].m2<<" "<<parameterek[savedI+j].m3<<" "<<parameterek[savedI+j].adasVeteliNapok<<" ";
            ofile<<parameterek[savedI+j].buy<<" "<<parameterek[savedI+j].toresAlatt<<" "<<parameterek[savedI+j].mi<<" "<<parameterek[savedI+j].ms<<" ";
            ofile<<parameterek[savedI+j].tores<<" ";
            ofile<<scores[j].atlagosNapiProfit<<" "<<scores[j].atlagosProfit<<" "<<scores[j].egyNapMaximum<<" ";//<<endl;
            for (int i=0;i<25;i++){
                ofile<<scores[j].alkalmakEvente[i]<<" ";
            }
            for (int i=0;i<25;i++){
                ofile<<scores[j].evVegiek[i]<<" ";
            }
            ofile<<endl;
        }

        ///cout<<endl<<endl<<"FOR"<<endl<<endl;

        ///Sleep(1);

    }

    /*Parameterek params; params.m1=13; params.m2=19; params.m3=49;
                            params.adasVeteliNapok=0;
                            params.tores=0.0039f;
                            cnt++;*/
                            //getScore(reszvenyek,params);
    cout<<"TIME: "<<(clock()-t)<<endl;
    //Sleep(1000);
    cout<<"TIME: "<<(clock()-t)<<endl;
    cout<<"CNT: "<<cnt<<" "<<parameterek.size()<<endl;





    return 0;
}
