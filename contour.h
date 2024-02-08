
#include <vector>
#include <iostream>
#include <fstream>
#include "json/json.h"

#ifndef CONTOUR_H
#define CONTOUR_H

struct line
{
    double x1,y1,x2,y2;
};

struct long_line
{
    std::vector<double> x,y;
};

class contour_line_unit
{
private:
    

public:
    int xbase,ybase;
    /**
     * vector of lines,
     * each line is just one straight line defined by 2 points.
    */
    std::vector< struct line> lines; 
    /**
     * vector of long lines,
     * each long line is a sequence of points.
    */
    std::vector<class long_line> flines;
    class long_line fline;


    contour_line_unit();
    ~contour_line_unit();
    friend std::ostream& operator<<(std::ostream& os, const contour_line_unit& u);

    inline void set_base(int i,int j){xbase=i;ybase=j;};
    inline void add_line(double x1, double y1, double x2, double y2)
    {
        struct line l;
        l.x1=x1;l.x2=x2;l.y1=y1;l.y2=y2;
        lines.push_back(l);
    }

    void run();
    void swap();
};


struct contour_line
{
    double level;
    std::vector<double> x,y;
};


class ccontour
{
private:
    int xdim,ydim;
    std::vector<double> z;

    /**
     * line_units[contour level index][line unit index]
    */
    std::vector< std::vector< class contour_line_unit> > line_units;  

    /**
     * lines[contour level index][line index]
     * Each contour_line is a closed contour line (or open if at the edge of the spectrum).
    */
    std::vector< std::vector<struct contour_line> > lines;

    int share_point(int,int,int);

public:
    ccontour();
    ~ccontour();

    /**
     * Marching squares algorithm part. 
     * x1,y1,x2,y2 are the signal region to speed up the contour calculation, because we do not need to check empty region.
    */
    bool conrec(
        std::vector<std::vector<double>> &d, /** data */
        const std::vector<double> &z_,       /** contour levels */
        const std::vector<int> x1,           /** signal region left*/
        const std::vector<int> y1,           /** signal region bottom*/
        const std::vector<int> x2,           /** signal region right*/
        const std::vector<int> y2            /** signal region top*/
    );

    void group_line();
    void print_intermedia();
    void print();
    void print_json(Json::Value &c,const double,const double,const double,const double);
    void save_result(Json::Value &infor, std::vector<float> &raw_data);
};

#endif // CONTOUR_H