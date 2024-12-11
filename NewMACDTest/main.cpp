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
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

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
    stringstream ss; ss<<"C:\\stockData\\daily\\groups\\"<<fname;
    std::unordered_set<std::string> stockNames = readStockNames(ss.str());

    std::vector<std::string> foundFiles;

    if (stockNames.empty()) {
        std::cerr << "Nincs elérhetõ részvény név." << std::endl;
        return foundFiles;
    }


    std::vector<std::string> directories = {
        "C:\\stockData\\daily\\us\\nasdaq stocks\\1",
        "C:\\stockData\\daily\\us\\nasdaq stocks\\2",
        "C:\\stockData\\daily\\us\\nasdaq stocks\\3",
        "C:\\stockData\\daily\\us\\nyse stocks\\1",
        "C:\\stockData\\daily\\us\\nyse stocks\\2"
    };

    // Végigmegyünk az összes mappán és megkeressük a fájlokat.
    for (const auto& dir : directories) {
        ///stringstream ss; ss<<folder<<dir;
        findStockFiles(stockNames, dir, foundFiles);
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

    Datum& operator=(const Datum& other) {
        if (this != &other) { // Önhozzárendelés ellenőrzése
            this->ev = other.ev;
            this->honap = other.honap;
            this->nap = other.nap;
        }
        return *this; // Láncolhatóvá tesszük
    }

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
    vector<int> mozgoatlagokDatum;
    string nev;
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
    //ifileStockData>>fejlec;
    getline(ifileStockData,toSplit);
    vector<string> lines; lines.reserve(12000);
    ///cout<<"OK"<<endl;
    int MOZGO_CNT = 50;
    while (!ifileStockData.eof()){
        getline(ifileStockData,toSplit);
        if (toSplit.size()>2)
            lines.push_back(toSplit);
    }
    ///cout<<"OK2"<<endl;
    vector<string> elemek;
    reszveny.mozgoatlag.resize(MOZGO_CNT);
    vector<list<MozgoAtlag>> mozgAtlgk(MOZGO_CNT);
    for (string str: lines){
        ///cout<<"OK3"<<endl;
        ///cout<<str<<endl;
        //ifileStockData>>toSplit;
        toSplit=str;
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

        for (float n=1.0f; n<=MOZGO_CNT; n++){
            if (reszveny.napok.size()<n) continue;
            if (reszveny.napok.size()==n) {
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
        }
    }
    elemek = split(elemek[0],".");
    reszveny.nev=elemek[0];
    ///cout<<reszveny.nev<<endl;
}

vector<Reszveny> reszvenyekParhuzamosBetoltese(vector<string> fajlnevek){
    vector<Reszveny> osszesReszveny; osszesReszveny.reserve(2000);
    int thCnt =4;
    vector<thread> szalak; szalak.resize(thCnt);
    vector<Reszveny> reszvenyek; reszvenyek.resize(thCnt);

    cout<<"CNT: "<<fajlnevek.size()<<endl;
    clock_t t1 = clock();
    clock_t t2 = clock();
    clock_t t3 = clock();
    for (size_t i=0; i<fajlnevek.size();){
        int savedI = i;
        t1=clock();
        for (int j=0; j<thCnt; j++){
            szalak[j] = thread(reszvenyBetoltese,ref(reszvenyek[j]),fajlnevek[i]);
            i++;
            if (i>=fajlnevek.size()) break;
        }
        //cout<<clock()-t1<<endl;

        float ts = 0;
        for (int j=0; j<thCnt; j++){
            if (savedI+j>=fajlnevek.size()) continue;
            clock_t t1 = clock();
            //if (szalak[j].joinable())
                szalak[j].join();
                //thread.
            if (clock()-t2>=5000){
                cout<<savedI+j<<" "<<clock()-t2<<" i "<<ts<<" - "<<(float)(clock()-t3)/(float)(savedI+j)<<endl;
                t2=clock();
            }
            osszesReszveny.push_back(reszvenyek[j]);
        }
        //cout<<clock()-t1<<endl;
    }
    return osszesReszveny;
}

struct Parameterek{
    int adasVeteliNapok = 0; /// (0-5)
    int m1 = 1, m2=2, m3=3; /// a három vizsgált mozgóátlag (1-50, különözőek)
    int ms = 0; /// mozgóátlagok növekvő sorrendje (0-5)
    int mi = 0; /// m1 és m3 növekszik / csökken
    bool buy = 0; /// adni vagy venni kell
    bool toresAlatt = 0; /// kisebb vaagy nagyobb legyen a törésnél?
    float tores = 0.0039f;

    bool operator==(const Parameterek& other) const {
        return (adasVeteliNapok==other.adasVeteliNapok &&
                m1==other.m1 &&
                m2==other.m2 &&
                m3==other.m3 &&
                ms==other.ms &&
                mi==other.mi &&
                buy==other.buy &&
                toresAlatt==other.toresAlatt// &&
                //tores==other.tores
                );
    }

};

bool chkSorrend(const float& a, const float& b, const float& c){
    return (a<=b && b<=c);
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

vector<ReszvenyGPU> getReszvenyekGPU(vector<Reszveny>& reszvenyek,vector<Datum>& osszesDatum){
    vector<ReszvenyGPU> ret;
    //vector<float> atlag(osszesDatum.size(),0);
    const int MOZGO_CNT = 50;
    for (int i=0; i<reszvenyek.size(); i++){
        if (i==383) continue; /// WKE
        cout<<"I "<<i<<" "<<reszvenyek[i].nev<<endl;
        ReszvenyGPU rgpu;
        rgpu.nev=reszvenyek[i].nev;
        rgpu.mozgoatlagokAtlag.resize(MOZGO_CNT,vector<float>(osszesDatum.size()));
        rgpu.mozgoatlagokZaras.resize(osszesDatum.size());
        rgpu.mozgoatlagokDatum.resize(osszesDatum.size());
        for (int z=0; z<osszesDatum.size(); z++){
            rgpu.mozgoatlagokZaras[z]=-1;
        }

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
                while (reszvenyek[i].mozgoatlag[j].datum[midx]<osszesDatum[z]){
                        ///if (i==469 && j==20 && z==137) cout<<"PEKING2"<<endl;
                    //z--;
                    midx++;
                    //continue;
                }
                if (osszesDatum[z]<reszvenyek[i].mozgoatlag[j].datum[midx]){
                    ///if (i==469 && j==20 && z==137) cout<<"PEKING1"<<endl;
                    rgpu.mozgoatlagokAtlag[j][z]=-1;
                    rgpu.mozgoatlagokDatum[z]=osszesDatum[z].toInt();
                    continue;
                }
                ///if (i==2 && j==25 && z==1425) cout<<"ALMA2"<<endl;
                ///cout<<"ok2"<<endl;
                ///if (i==469 && j==20 && z==137) cout<<"PEKING3"<<endl;
                rgpu.mozgoatlagokAtlag[j][z]=reszvenyek[i].mozgoatlag[j].atlag[midx];
                rgpu.mozgoatlagokZaras[z]=reszvenyek[i].mozgoatlag[j].zaras[midx];
                rgpu.mozgoatlagokDatum[z]=reszvenyek[i].mozgoatlag[j].datum[midx].toInt();
                ///osszesDatum[z].toInt();
                ///if (reszvenyek[i].nev=="AMD" && j==0 && rgpu.mozgoatlagokDatum[z]==osszesDatum[z].toInt()) cout<<"OK"<<endl;
                ///if (reszvenyek[i].nev=="AMD" && j==0)cout<<(int)osszesDatum[z].toInt()<<" 1"<<endl;
                ///if (reszvenyek[i].nev=="AMD" && j==0)cout<<(int)rgpu.mozgoatlagokDatum[z]<<"   2"<<endl;
                ///if (reszvenyek[i].nev=="AMD" && j==0)cout<<(int)osszesDatum[z].toInt()-(int)rgpu.mozgoatlagokDatum[z]<<"   2"<<endl;
                midx++;
            }
        }
        ret.push_back(rgpu);
    }

    return ret;
}

void reszvenyekJavitasa(vector<Reszveny>& reszvenyek){
    for (int i=0; i<reszvenyek.size(); i++){
        for (int j=1; j<reszvenyek[i].napok.size()-1; j++){
            float a1 = reszvenyek[i].napok[j-1].nyitas, a2 = reszvenyek[i].napok[j].nyitas, a3 = reszvenyek[i].napok[j+1].nyitas;
            if (a2*10.f<a1+a3 || a2/10.f>a1+a3) cout<<reszvenyek[i].nev<<endl;
        }
    }
    for (int i=0; i<reszvenyek.size(); i++){
        for (int j=1; j<reszvenyek[i].napok.size()-1; j++){
            float a1 = reszvenyek[i].napok[j-1].zaras, a2 = reszvenyek[i].napok[j].zaras, a3 = reszvenyek[i].napok[j+1].zaras;
            //cout<<reszvenyek[i].nev<<endl;
            if (a2*10.f<a1+a3 || a2/10.f>a1+a3) cout<<reszvenyek[i].nev<<endl;
        }
    }
    for (int i=0; i<reszvenyek.size(); i++){
        for (int j=1; j<reszvenyek[i].napok.size()-1; j++){
            float a1 = reszvenyek[i].napok[j-1].minimum, a2 = reszvenyek[i].napok[j].minimum, a3 = reszvenyek[i].napok[j+1].minimum;
            //cout<<reszvenyek[i].nev<<endl;
            if (a2*10.f<a1+a3 || a2/10.f>a1+a3) cout<<reszvenyek[i].nev<<endl;
        }
    }
    for (int i=0; i<reszvenyek.size(); i++){
        for (int j=1; j<reszvenyek[i].napok.size()-1; j++){
            float a1 = reszvenyek[i].napok[j-1].maximum, a2 = reszvenyek[i].napok[j].maximum, a3 = reszvenyek[i].napok[j+1].maximum;
            //cout<<reszvenyek[i].nev<<endl;
            if (a2*10.f<a1+a3 || a2/10.f>a1+a3) cout<<reszvenyek[i].nev<<endl;
        }
    }
}

struct Pelda{
    int stockId;
    int napId;
    ///float aznap;
    ///float masnap;
    ///float harmadnap;
};

struct Eset{
    int stockId = -1;
    float prod = -1;
};

struct NapiEset{
    vector<Eset> esetek;
    float cnt=0;
    float prod=0;
};

struct Vizsgalt{
    Parameterek param;
    vector<Pelda> peldak;
};

int main(){

    clock_t fullT = clock();
    vector<string> reszvenyekFajlNeve = reszvenyekEleresiUtja("errorLessEtoro.txt","data");
    vector<Reszveny> reszvenyek = reszvenyekParhuzamosBetoltese(reszvenyekFajlNeve);
    vector<Datum> osszesDatum = getOsszesDatum(reszvenyek);
    clock_t time0 = clock();
    cout<<"DONE "<<osszesDatum.size()<<endl;
    reszvenyekJavitasa(reszvenyek);
    vector<ReszvenyGPU> reszvenyekGpu = getReszvenyekGPU(reszvenyek, osszesDatum);

    vector<Vizsgalt> vizsglatak;
    for (int m1=2; m1<21; m1++){
        Vizsgalt vTemp;
        for (int m2=m1+1; m2<50; m2++){
            for (int m3=m2+1; m3<50; m3++){
                Parameterek param; param.m1=m1; param.m2=m2; param.m3=m3;
                for (int i=0; i<6; i++){
                    for (int j=0; j<4; j++){
                        param.ms=i; param.mi=j;
                        for (float k=-0.01f; k<0.01f; k+=0.0001f){
                            if (k>-0.0001f && k<0.0001f) k=0.0f;
                            param.tores=k; vTemp.param=param; vTemp.peldak.reserve(100000);
                            vizsglatak.push_back(vTemp);
                        }
                    }
                }
            }
        }
    }
    cout<<"ALMA"<<endl;
    ///Sleep(1000);


    long long indCnt = 0;
    long long endCnt = 1;
    long long cnt2=0;
    long long endCnt2=1;
    ///Eset eset;
    ofstream ofile("eredmenyek.txt");
    bool endC = true;
    for (int m1=2; m1<21 && endC; m1++){
        for (int m2=m1+1; m2<50 && endC; m2++){
            for (int m3=m2+1; m3<50 && endC; m3++){
                if (endCnt2<=cnt2) break;
                cout<<cnt2++<<endl;
                ///if (indCnt!=endCnt) indCnt++;
                ///else {endC=false; break;}
                ///eset.parameterek.m1=m1;
                ///eset.parameterek.m2=m2;
                ///eset.parameterek.m3=m3;
                for (int b=0; b<2 && endC; b++){
                    for (int ad=0; ad<2 && endC; ad++){
                        for (float tk=-0.01f; tk<0.01f && endC; tk+=0.0002f){
                            cout<<"TK: "<<tk<<endl;
                            if (tk>-0.0001f && tk<0.0001f) tk=0.0f;
                            for (int sor=0; sor<6; sor++){
                                for (int tend=0; tend<8;tend++){
                                    for (int ta=0; ta<2; ta++) {


                                        vector<NapiEset> napiEsetek; napiEsetek.reserve(osszesDatum.size());
                                        for (int j=2; j<osszesDatum.size()-2; j++){
                                            vector<Eset> esetek;
                                            //cout<<"I: "<<i<<" - "<<esetek.size()<<endl;
                                            for (int i=0; i<reszvenyekGpu.size(); i++){
                                                //cout<<"J: "<<j<<endl;
                                                ///eset.nev=reszvenyekGpu[i].nev; eset.stockId=i;
                                                float ma11 = reszvenyekGpu[i].mozgoatlagokAtlag[m1][j-2];
                                                float ma21 = reszvenyekGpu[i].mozgoatlagokAtlag[m1][j-1];
                                                float ma31 = reszvenyekGpu[i].mozgoatlagokAtlag[m1][j];
                                                if (ma11==-1 || ma21==-1 || ma31==-1) continue;

                                                float ma12 = reszvenyekGpu[i].mozgoatlagokAtlag[m2][j-2];
                                                float ma22 = reszvenyekGpu[i].mozgoatlagokAtlag[m2][j-1];
                                                float ma32 = reszvenyekGpu[i].mozgoatlagokAtlag[m2][j];
                                                float ma42 = reszvenyekGpu[i].mozgoatlagokAtlag[m2][j+1];
                                                float ma52 = reszvenyekGpu[i].mozgoatlagokAtlag[m2][j+2];
                                                if (ma12==-1 || ma22==-1 || ma32==-1 || ma42==-1 || ma52==-1) continue;

                                                float ma13 = reszvenyekGpu[i].mozgoatlagokAtlag[m3][j-2];
                                                float ma23 = reszvenyekGpu[i].mozgoatlagokAtlag[m3][j-1];
                                                float ma33 = reszvenyekGpu[i].mozgoatlagokAtlag[m3][j];
                                                if (ma13==-1 || ma23==-1 || ma33==-1) continue;

                                                ///eset.aznapi=reszvenyekGpu[i].mozgoatlagokZaras[j];
                                                ///eset.masnapi=reszvenyekGpu[i].mozgoatlagokZaras[j+1];
                                                ///eset.harmadnap=reszvenyekGpu[i].mozgoatlagokZaras[j+2];

                                                ///cout<<"i2: "<<i<<endl;

                                                float b1 = ma22-ma12, b2 = ma32-ma22;
                                                int elojel = 0; /// pozitív
                                                //if (b2<0 && b1<0 && b1<b2) elojel=1; /// negatív
                                                //else if(b2>0 && b1>0 && b1>b2) elojel=0;
                                                if (b2<0 && b1<0) elojel=1; /// negatív
                                                else if(b2>0 && b1>0) elojel=0;
                                                else continue;

                                                float tores = (b1-b2)/ma22;
                                                ///eset.parameterek.tores = tores;
                                                if (ta==0) if (tores<tk) continue;
                                                if (ta==1) if (tores>tk) continue;
                                                if (tores==0) continue;

                                                int sorrend = 0;
                                                if (chkSorrend(ma33,ma32,ma31)) sorrend = 0;
                                                else if (chkSorrend(ma33,ma31,ma32)) sorrend = 1;
                                                else if (chkSorrend(ma32,ma31,ma33)) sorrend = 2;
                                                else if (chkSorrend(ma32,ma33,ma31)) sorrend = 3;
                                                else if (chkSorrend(ma31,ma33,ma32)) sorrend = 4;
                                                else if (chkSorrend(ma31,ma32,ma33)) sorrend = 5;
                                                if (sor!=sorrend) continue;
                                                ///eset.parameterek.ms = sorrend;

                                                int tendencia = 0;
                                                if (ma32-ma22 <0) tendencia+=1;
                                                if (ma33-ma23 <0) tendencia+=2;
                                                if (ma31-ma21 <0) tendencia+=4;
                                                if (tend!=tendencia) continue;
                                                ///eset.parameterek.mi = tendencia;

                                                //if ()
                                                Eset eset;eset.stockId=i;
                                                if (b==1 && ad == 0) eset.prod=ma42/ma32;
                                                else if (b==0 && ad == 0) eset.prod=2.0f-ma42/ma32;
                                                else if (b==1 && ad == 1) eset.prod=ma52/ma42;
                                                else if (b==0 && ad == 1) eset.prod=2.0f-ma52/ma42;

                                                esetek.push_back(eset);
                                            }
                                            NapiEset neTemp;
                                            float s = 0;
                                            for (int k=0; k<esetek.size(); k++){
                                                neTemp.esetek.push_back(esetek[k]);
                                                neTemp.prod+=esetek[k].prod-1.0f;
                                                neTemp.cnt++;
                                                s++;
                                            }
                                            if (s!=0)
                                                neTemp.prod/=s;
                                            napiEsetek.push_back(neTemp);
                                        }

                                        ofile<<indCnt<<" "<<m1<<" "<<m2<<" "<<m3<<" "<<tk<<" "<<ta<<" "<<sor<<" "<<tend<<" "<<b<<" "<<ad<<endl;
                                        float effhoz=100.f;
                                        for (int k=0; k<napiEsetek.size(); k++){
                                            ofile<<effhoz<<" ";
                                            effhoz*=napiEsetek[k].prod+1.0f;
                                        }
                                        ofile<<endl;
                                        for (int k=0; k<napiEsetek.size(); k++){
                                            ofile<<napiEsetek[k].cnt<<" ";
                                        }
                                        ofile<<endl;
                                        indCnt++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }



    return 0;
}
