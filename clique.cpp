// Program for Floyd Warshall Algorithm
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <algorithm>

#include "clique.h"

//#define MY_DEBUG

cmaxclique::cmaxclique(){}
cmaxclique::~cmaxclique(){}



int cmaxclique::pivot(std::vector<int> input)
{

    int index;
    int m=0;
    
    for(int i=0;i<input.size();i++)
    {
        if(nneighbor[input[i]]>m)
        {
            m=nneighbor[input[i]];
            index=i;
        }
    }
    return input[index];
}



void cmaxclique::bk1(int c,std::vector<int> r, std::vector<int> p, std::vector<int> x)
{
    std::vector<int> r0,p0,x0;
    std::vector<int> t_int;
    c=c+1;
    if(p.size()==0 && x.size()==0)
    {
        //cout<<counter++<<": Level "<<c<<" Max clique: ";
        for(int i=0;i<r.size();i++)
        {
            //cout<<r[i]+1<<" ";
            t_int.push_back(r[i]);
        }
        //cout<<endl;
        clique.push_back(t_int);
        return;
    }
    

    for(int i=p.size()-1;i>=0;i--)
    {
        r0=r;
        r0.push_back(p[i]);
        
        p0.clear();
        for(int j=0;j<p.size();j++)
        {
            if(neighbor[p[j]*npeak+p[i]]==1)
                p0.push_back(p[j]);
        }
        
        x0.clear();
        for(int j=0;j<x.size();j++)
        {
            if(neighbor[x[j]*npeak+p[i]]==1)
                x0.push_back(x[j]);
        }
        
        
        bk1(c,r0,p0,x0);
        x.push_back(p[p.size()-1]);
        p.pop_back();
    }
    
}



void cmaxclique::bk2(int c,std::vector<int> r, std::vector<int> p, std::vector<int> x)
{
    std::vector<int> r0,p0,x0;
    c=c+1;
    if(p.size()==0 && x.size()==0 )
    {
        if(r.size()>=2)
        {
            std::vector<int> t_int;
            //std::cout<<counter++<<": Level "<<c<<" Max clique: ";
            for(int i=0;i<r.size();i++)
            {
                //std::cout<<r[i]+1<<" ";
                t_int.push_back(r[i]);
            }
            //std::cout<<std::endl;
            clique.push_back(t_int);
        }
        return;
    }
    
   
    std::vector<int> pp;
    pp=p;
    pp.insert( pp.end(), x.begin(), x.end() );
    
    int index=pivot(pp);
    pp=p;
    for(int i=pp.size()-1;i>=0;i--)
    {
        if(neighbor[index*npeak+pp[i]]==1)
            pp.erase(pp.begin()+i);
    }
    
    for(int i=pp.size()-1;i>=0;i--)
    {
        r0=r;
        r0.push_back(pp[i]);
        
        p0.clear();
        for(int j=0;j<p.size();j++)
        {
            if(neighbor[p[j]*npeak+pp[i]]==1)
                p0.push_back(p[j]);
        }
        
        x0.clear();
        for(int j=0;j<x.size();j++)
        {
            if(neighbor[x[j]*npeak+pp[i]]==1)
                x0.push_back(x[j]);
        }
        
#ifdef MY_DEBUG
        for(int i=0;i<c-2;i++)
            std::cout<<" ";
        std::cout<<c<<" r:";
        for(int i=0;i<r.size();i++)
            std::cout<<r[i]+1<<" ";
        std::cout<<"p: ";
        for(int i=0;i<p.size();i++)
            std::cout<<p[i]+1<<" ";
        std::cout<<"x: ";
        for(int i=0;i<x.size();i++)
            std::cout<<x[i]+1<<" ";
        std::cout<<std::endl;
#endif
        
        bk2(c,r0,p0,x0);
        x.push_back(pp[i]);
        
        p.erase(std::remove(p.begin(), p.end(), pp[i]), p.end());
    }

}

void cmaxclique::init(std::vector< std::vector< int> > *n)
{
    npeak=n->size();

    neighbor.clear();
    for(int i=0;i<npeak;i++)
    {
        neighbor.insert(neighbor.end(),n->at(i).begin(),n->at(i).end());
    }

    for(int i=0;i<npeak;i++)
    {
        neighbor[i*npeak+i]=0;
        int c=0;
        for(int j=0;j<npeak;j++)
            c+=neighbor[i*npeak+j];
        nneighbor.push_back(c);
    }

    // std::cout<<"In cmaxclique, npeak is "<<npeak<<" and neighbor is: "<<std::endl;
    // for(int i=0;i<npeak*npeak;i++) std::cout<<neighbor[i]<<" ";
    // std::cout<<std::endl;


    return;
}




void cmaxclique::solver()
{
    std::vector<int> r,p,x;
    r.clear();
    p.clear();
    x.clear();
    
    for(int i=0;i<npeak;i++)
        p.push_back(i);
    counter=0;
    bk2(1,r,p,x);

}



void cmaxclique::add_orphan()
{
    //all orphan node are clique by itself!
    std::vector<int> flags;
    flags.resize(npeak, 0);
    
    for (int i = 0; i < clique.size(); i++)
    {
        for (int j = 0; j < clique[i].size(); j++)
        {
            flags[clique[i][j]] = 1;
        }
    }

    for (int i = 0; i < npeak; i++)
    {
        if (flags[i] == 0)
        {
            std::vector<int> t;
            t.push_back(i); 
            clique.push_back(t);
        }
    }
}


std::vector< std::vector<int> > cmaxclique::output()
{
    // std::cout<<"In cmaxlcique, Cliques are:"<<std::endl;
    // for(int i=0;i<clique.size();i++)
    // {
    //     for(int j=0;j<clique[i].size();j++)
    //     {
    //         std::cout<<clique[i][j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<"**************************"<<std::endl;



    return clique;
}

