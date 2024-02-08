
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "json/json.h"
#include "contour.h"

#define xsect(p1, p2) (h[p2] * xh[p1] - h[p1] * xh[p2]) / (h[p2] - h[p1])
#define ysect(p1, p2) (h[p2] * yh[p1] - h[p1] * yh[p2]) / (h[p2] - h[p1])


//class contour_line_unit
std::ostream& operator<<(std::ostream& os, const contour_line_unit& u)
{
    os << u.xbase<<" "<<u.ybase<<" ";
    for(int i=0;i<u.fline.x.size();i++)
    {
        os<<u.fline.x[i]<<" "<<u.fline.y[i]<<" ";
    }
    return os;
}

contour_line_unit::contour_line_unit(){};
contour_line_unit::~contour_line_unit(){};


void contour_line_unit::swap()
{
    std::reverse(fline.x.begin(),fline.x.end()); 
    std::reverse(fline.y.begin(),fline.y.end()); 
};


void contour_line_unit::run()
{
    int nl=lines.size();

    std::vector< std::vector<int> > neighbor(nl,std::vector<int>(nl,0));
    std::vector<int> nn(nl,0);

    for(int i=0;i<nl;i++)
    {
        for(int j=i+1;j<nl;j++)
        {
            if(lines[i].x1==lines[j].x1 && lines[i].y1==lines[j].y1)
            {
                neighbor[i][j]=1;
                neighbor[j][i]=1;
            }
            else if(lines[i].x1==lines[j].x2 && lines[i].y1==lines[j].y2)
            {
                neighbor[i][j]=2;
                neighbor[j][i]=3;
            }
            else if(lines[i].x2==lines[j].x1 && lines[i].y2==lines[j].y1)
            {
                neighbor[i][j]=3;
                neighbor[j][i]=2;
            }
            else if(lines[i].x2==lines[j].x2 && lines[i].y2==lines[j].y2)
            {
                neighbor[i][j]=4;
                neighbor[j][i]=4;
            }
        }
    }
    // std::cout<<"nl is "<<nl<<" finish neighbor. Size of nn is "<<nn.size()<<std::endl;

    // std::cout<<"Neighbor matirx is:"<<std::endl;

    //  for(int i=0;i<nl;i++)
    // {
    //     for(int j=0;j<nl;j++)
    //     {
    //         std::cout<<neighbor[i][j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }


    std::vector<int> used(nl, 0);
    for (int i = 0; i < nl; i++)
    {
        if(used[i]==1) continue;
        for (int j = 0; j < nl; j++)
        {
            if (neighbor[i][j] > 0)
            {
                nn[i]++;
            }
        }
        if (nn[i] == 1)
        {
            std::vector<int> order;
            std::vector<int> order_type;

            order.clear();
            order_type.clear();

            order.push_back(i);
            order_type.push_back(0);
            //std::cout<<"put "<<i<<" into order and put "<<0<<" into order type."<<std::endl;

            int current_p = 0;
            while (order.size() > current_p)
            {
                int t = order[current_p];
                used[t] = 1;
                for (int i = 0; i < nl; i++)
                {
                    if (used[i] == 1)
                        continue;
                    if (neighbor[t][i] > 0)
                    {
                        order.push_back(i);
                        order_type.push_back(neighbor[t][i]);
                        break;
                    }
                }
                current_p++;
            }

            // if(nl!=order.size()) //not all line segment are connect, sth is wrong
            // {
            //     std::cout<<"nl="<<nl<<" but size of order is "<<order.size()<<" and "<<order_type.size()<<std::endl;
            //     std::cout<<"line segments are:"<<std::endl;
            //     for(int i=0;i<nl;i++)
            //     {
            //         std::cout<<lines[i].x1<<" "<<lines[i].y1<<" "<<lines[i].x2<<" "<<lines[i].y2<<std::endl;
            //     }
            //     std::cout<<"Connected segments are:"<<std::endl;
            //     for(int i=0;i<order.size();i++)
            //     {
            //         std::cout<<order[i]<<" ";
            //     }
            //     std::cout<<std::endl;
            // }

            for (int i = 1; i < order.size(); i++)
            {
                int t = order[i - 1];
                int t2 = order[i];
                int tp = order_type[i];

                if (tp == 1)
                {
                    std::swap(lines[t].x1, lines[t].x2);
                    std::swap(lines[t].y1, lines[t].y2);
                }
                else if (tp == 2)
                {
                    std::swap(lines[t].x1, lines[t].x2);
                    std::swap(lines[t].y1, lines[t].y2);
                    std::swap(lines[t2].x1, lines[t2].x2);
                    std::swap(lines[t2].y1, lines[t2].y2);
                    if (i < order.size() - 1)
                    {
                        if (order_type[i + 1] == 1)
                            order_type[i + 1] = 3;
                        else if (order_type[i + 1] == 2)
                            order_type[i + 1] = 4;
                        else
                            std::cout << "ERROR in contour_line_unit::run()" << std::endl;
                    }
                }
                else if (tp == 4)
                {
                    std::swap(lines[t2].x1, lines[t2].x2);
                    std::swap(lines[t2].y1, lines[t2].y2);
                    if (i < order.size() - 1)
                    {
                        if (order_type[i + 1] == 1)
                            order_type[i + 1] = 3;
                        else if (order_type[i + 1] == 2)
                            order_type[i + 1] = 4;
                        else
                            std::cout << "ERROR in contour_line_unit::run()" << std::endl;
                    }
                }
            }

            class long_line fline;
            fline.x.push_back(lines[order[0]].x1);
            fline.y.push_back(lines[order[0]].y1);
            for (int i = 1; i < order.size(); i++)
            {
                fline.x.push_back(lines[order[i]].x2);
                fline.y.push_back(lines[order[i]].y2);
            }
            flines.push_back(fline);
        }//if (nn[i] == 1)
    }





};


// class ccontour
ccontour::ccontour(){};
ccontour::~ccontour(){};

void ccontour::print_intermedia()
{
    for(int i=0;i<line_units.size();i++)
    {
        for(int j=0;j<line_units[i].size();j++)
        {
            std::cout<<line_units[i][j]<<std::endl;
        }
    }
};

int ccontour::share_point(int ndx, int i, int j)
{
    int result=0;
    class contour_line_unit u1;
    class contour_line_unit u2;
    u1=line_units[ndx][i];
    u2=line_units[ndx][j];
    int npoint1=u1.fline.x.size();
    int npoint2=u2.fline.x.size();
    
    if(u1.fline.x[0]==u2.fline.x[0] && u1.fline.y[0]==u2.fline.y[0])
    {
        result=1;
    }
    else if(u1.fline.x[0]==u2.fline.x[npoint2-1] && u1.fline.y[0]==u2.fline.y[npoint2-1])
    {
        result=2;
    }
    else if(u1.fline.x[npoint1-1]==u2.fline.x[0] && u1.fline.y[npoint1-1]==u2.fline.y[0])
    {
        result=3;
    }
    else if(u1.fline.x[npoint1-1]==u2.fline.x[npoint2-1] && u1.fline.y[npoint1-1]==u2.fline.y[npoint2-1])
    {
        result=4;
    }
    else
    {
        result=0;
    }
    return result;
}

void ccontour::group_line()
{
    int table[5]={0,1,3,2,4};
    for(int ii=0;ii<line_units.size();ii++)
    {
        //work on line_units[ii] here
        //some line line_units contain more than one fline!!, if so, seperate them
        std::vector< class contour_line_unit> t_lines_one_level=line_units[ii];
        std::vector< class contour_line_unit> lines_one_level;

        for(int i=0;i<t_lines_one_level.size();i++)
        {
            if(t_lines_one_level[i].flines.size()==1)
            {
                t_lines_one_level[i].fline=t_lines_one_level[i].flines[0];
                lines_one_level.push_back(t_lines_one_level[i]);
            }
            for(int j=1;j<t_lines_one_level[i].flines.size();j++)
            {
                class contour_line_unit temp;
                temp.xbase= t_lines_one_level[i].xbase;
                temp.ybase= t_lines_one_level[i].ybase;
                temp.fline= t_lines_one_level[i].flines[j];   
                lines_one_level.push_back(temp);
            }
        }
        line_units[ii]=lines_one_level;

        int nline=lines_one_level.size();
        std::vector< std::vector<int> > p(xdim,std::vector<int>(ydim,-1));        

        for(int j=0;j<lines_one_level.size();j++)
        {
            p[lines_one_level[j].xbase][lines_one_level[j].ybase]=j;
        }

        std::vector< struct contour_line > tlines;
        for(int m0=1;m0<xdim-1;m0++)
        {
            for(int n0=1;n0<ydim-1;n0++)
            {
                if(p[m0][n0]>=0)
                {
                    std::vector<int> mm,nn;
                    std::vector<int> g;
                    std::vector<int> order;
                    int nchecked=0;
                    mm.push_back(m0);
                    nn.push_back(n0);
                    g.push_back(p[m0][n0]);
                    order.push_back(0);
                  
                    while(mm.size()>nchecked)
                    {
                        int t;
                        int m=mm[nchecked];
                        int n=nn[nchecked];
                        //std::cout<<"m="<<m<<",n="<<n<<std::endl;
                        //std::cout<<"share_point m,n and m,n+1 rerurn "<<share_point(i,p[m][n],p[m][n+1])<<std::endl;
                        nchecked++;
                        if(p[m+1][n]>=0 && (t=share_point(ii,p[m][n],p[m+1][n]))>0)
                        {
                           mm.push_back(m+1);
                           nn.push_back(n);
                           g.push_back(p[m+1][n]);
                           order.push_back(t);
                        }
                        else if(p[m-1][n]>=0 && (t=share_point(ii,p[m][n],p[m-1][n]))>0)
                        {
                           mm.push_back(m-1);
                           nn.push_back(n);
                           g.push_back(p[m-1][n]);
                           order.push_back(t);
                        }
                        else if(p[m][n+1]>=0 && (t=share_point(ii,p[m][n],p[m][n+1]))>0)
                        {
                           mm.push_back(m);
                           nn.push_back(n+1);
                           g.push_back(p[m][n+1]);
                           order.push_back(t);
                        }
                        else if(p[m][n-1]>=0 && (t=share_point(ii,p[m][n],p[m][n-1]))>0)
                        {
                           mm.push_back(m);
                           nn.push_back(n-1);
                           g.push_back(p[m][n-1]);
                           order.push_back(t);
                        }
                        p[m][n]=-1;
                        //std::cout<<"Szie of mm is "<<mm.size()<<" ncheck is "<<nchecked<<std::endl;
                    }

                    //we found one contour line here
                    // std::cout<<"Found one contour line:"<<std::endl;
                    // for(int k=0;k<g.size();k++)
                    // {
                    //     std::cout<<lines_one_level[g[k]]<<std::endl;
                    // }
                    // std::cout<<std::endl;

                    int nl=g.size();
                    for(int i=1;i<nl;i++)
                    {
                        int t=g[i-1];
                        int t2=g[i];
                        int tp=order[i];

                        if(tp==1)
                        {
                            lines_one_level[t].swap();
                        }
                        else if(tp==2)
                        {
                            lines_one_level[t].swap();
                            lines_one_level[t2].swap();
                            if(i<nl-1)
                            {
                                if(order[i+1]==1) order[i+1]=3;
                                else if (order[i+1]==2) order[i+1]=4;
                                else std::cout<<"ERROR in ccontour::group_line()"<<std::endl;
                            }
                        }
                        else if(tp==4)
                        {
                            lines_one_level[t2].swap();
                            if(i<nl-1)
                            {
                                if(order[i+1]==1) order[i+1]=3;
                                else if (order[i+1]==2) order[i+1]=4;
                                else std::cout<<"ERROR in ccontour::group_line()"<<std::endl;
                            }
                        }
                    }

                    // std::cout<<"After ordering and reverse:"<<std::endl;
                    // for(int k=0;k<g.size();k++)
                    // {
                    //     std::cout<<lines_one_level[g[k]]<<std::endl;
                    // }
                    // std::cout<<std::endl;

                    struct contour_line tline;

                    tline.level=z[ii];
                    tline.x.insert(tline.x.end(),lines_one_level[g[0]].fline.x.begin(),lines_one_level[g[0]].fline.x.end());
                    tline.y.insert(tline.y.end(),lines_one_level[g[0]].fline.y.begin(),lines_one_level[g[0]].fline.y.end());
                    for(int k=1;k<g.size();k++)
                    {
                        tline.x.insert(tline.x.end(),lines_one_level[g[k]].fline.x.begin()+1,lines_one_level[g[k]].fline.x.end());
                        tline.y.insert(tline.y.end(),lines_one_level[g[k]].fline.y.begin()+1,lines_one_level[g[k]].fline.y.end());
                    }
                    tlines.push_back(tline);

                }//if(p[m0][n0]>=0)
            }//for(int n0=1;n0<ydim-1;n0++)
        }//for(int m0=1;m0<xdim-1;m0++)
        lines.push_back(tlines);
        p.clear();
    }
    return;
}

bool ccontour::conrec(
                        std::vector<std::vector<double>> &d, /** data */
                        const std::vector<double> &z_, /** contour levels */
                        const std::vector<int> signal_x1, /** signal region left*/
                        const std::vector<int> signal_y1, /** signal region bottom*/
                        const std::vector<int> signal_x2, /** signal region right*/
                        const std::vector<int> signal_y2 /** signal region top*/
                    )
{
    int m1, m2, m3, case_value;
    double dmin, dmax;
    int m;
    double h[5], x1, y1, x2, y2;
    int sh[5];
    double xh[5], yh[5];
    int im[4] = {0, 1, 1, 0};
    int jm[4] = {0, 0, 1, 1};
    int castab[3][3][3] = {{{0, 0, 8}, {0, 2, 5}, {7, 6, 9}},
                           {{0, 3, 4}, {1, 3, 1}, {4, 3, 0}},
                           {{9, 6, 7}, {5, 2, 0}, {8, 0, 0}}};

    xdim = d.size();
    ydim = d[0].size();
    z = z_;

    /**
     * Number of contour levels.
    */
    int nc = z.size();


    line_units.resize(nc);

    /**
     * Loop over contour levels.
    */
    for (int k = 0; k < nc; k++)
    {
        /**
         * Loop over all signal region.
         */
        for (int n = 0; n < signal_x1.size(); n++)
        {
            /**
             * loop over y dim of this signal region.
             */
            for (int j = signal_y1[n]; j <= signal_y2[n]; j++)
            {
                /**
                 * loop over x dim of this signal region.
                */
                for(int i = signal_x1[n]; i <= signal_x2[n]; i++)
                {
                    double temp1, temp2;
                    temp1 = std::min(d[i][j], d[i][j + 1]);
                    temp2 = std::min(d[i + 1][j], d[i + 1][j + 1]);
                    dmin = std::min(temp1, temp2);
                    temp1 = std::max(d[i][j], d[i][j + 1]);
                    temp2 = std::max(d[i + 1][j], d[i + 1][j + 1]);
                    dmax = std::max(temp1, temp2);

                    if (z[k] >= dmin && z[k] <= dmax)
                    {
                        class contour_line_unit line_unit;
                        line_unit.set_base(i, j);

                        for (m = 4; m >= 0; m--)
                        {
                            if (m > 0)
                            {
                                h[m] = d[i + im[m - 1]][j + jm[m - 1]] - z[k];
                                xh[m] = 1 * (i + im[m - 1]) + 0;
                                yh[m] = 1 * (j + jm[m - 1]) + 0;
                            }
                            else
                            {
                                h[0] = 0.25 * (h[1] + h[2] + h[3] + h[4]);
                                xh[0] = 0.5 * (i + i + 1) * 1 + 0;
                                yh[0] = 0.5 * (j + j + 1) * 1 + 0;
                            }
                            if (h[m] > 0.0)
                            {
                                sh[m] = 1;
                            }
                            else if (h[m] < 0.0)
                            {
                                sh[m] = -1;
                            }
                            else
                                sh[m] = 0;
                        }
                        for (m = 1; m <= 4; m++)
                        {
                            m1 = m;
                            m2 = 0;
                            if (m != 4)
                                m3 = m + 1;
                            else
                                m3 = 1;
                            case_value = castab[sh[m1] + 1][sh[m2] + 1][sh[m3] + 1];
                            if (case_value != 0)
                            {
                                switch (case_value)
                                {
                                case 1:
                                    x1 = xh[m1];
                                    y1 = yh[m1];
                                    x2 = xh[m2];
                                    y2 = yh[m2];
                                    break;
                                case 2:
                                    x1 = xh[m2];
                                    y1 = yh[m2];
                                    x2 = xh[m3];
                                    y2 = yh[m3];
                                    break;
                                case 3:
                                    x1 = xh[m3];
                                    y1 = yh[m3];
                                    x2 = xh[m1];
                                    y2 = yh[m1];
                                    break;
                                case 4:
                                    x1 = xh[m1];
                                    y1 = yh[m1];
                                    x2 = xsect(m2, m3);
                                    y2 = ysect(m2, m3);
                                    break;
                                case 5:
                                    x1 = xh[m2];
                                    y1 = yh[m2];
                                    x2 = xsect(m3, m1);
                                    y2 = ysect(m3, m1);
                                    break;
                                case 6:
                                    x1 = xh[m3];
                                    y1 = yh[m3];
                                    x2 = xsect(m1, m2);
                                    y2 = ysect(m1, m2);
                                    break;
                                case 7:
                                    x1 = xsect(m1, m2);
                                    y1 = ysect(m1, m2);
                                    x2 = xsect(m2, m3);
                                    y2 = ysect(m2, m3);
                                    break;
                                case 8:
                                    x1 = xsect(m2, m3);
                                    y1 = ysect(m2, m3);
                                    x2 = xsect(m3, m1);
                                    y2 = ysect(m3, m1);
                                    break;
                                case 9:
                                    x1 = xsect(m3, m1);
                                    y1 = ysect(m3, m1);
                                    x2 = xsect(m1, m2);
                                    y2 = ysect(m1, m2);
                                    break;
                                default:
                                    break;
                                }
                                line_unit.add_line(x1, y1, x2, y2);
                            }
                        }
                        /**
                         * This function call will generate flines from lines in class contour_line_unit.
                         */
                        line_unit.run();
                        line_units[k].push_back(line_unit);
                    } // if (dmax >= z[0] && dmin <= z[nc - 1])

                } // for i
            }// for j
        }
    } // for k
    return 0;
};

void ccontour::print()
{
    for(int i=0;i<lines.size();i++)
    {
        for(int j=0;j<lines[i].size();j++)
        {
            std::cout<<"level is "<<lines[i][j].level<<std::endl;
            for(int k=0;k<lines[i][j].x.size();k++)
            {
                std::cout<<lines[i][j].x[k]<<" "<<lines[i][j].y[k]<<std::endl;
            }
            std::cout<<std::endl;
        }
    }
};

void ccontour::save_result(Json::Value &infor, std::vector<float> &raw_data)
{
    /**
     * Play safe
    */
    infor.clear();
    raw_data.clear();

    /**
     * Write z to josn object as levels.
    */
    Json::Value contour_levels=Json::arrayValue;
    for(int i=0;i<z.size();i++)
    {
        contour_levels[i]=z[i];
    }
    infor["contour_levels"]=contour_levels;

    /**
     * Write contour to raw_data, Track the length of each contour line and length at each level.
    */
    std::vector<int> polygen_length,level_length;
    level_length.clear();
    polygen_length.clear();

    int n_previous = 0;
    /**
     * Loop over all levels.
    */
    for(int i=0;i<lines.size();i++)
    {
        /**
         * Loop over all contour lines at this level.
        */
        for(int j=0;j<lines[i].size();j++)
        {
            /**
             * Write one (closed except at edge) contour line to raw_data.
            */
            for(int m=0;m<lines[i][j].x.size();m++)
            {
                raw_data.push_back(lines[i][j].x[m]);
                raw_data.push_back(lines[i][j].y[m]);
            }
            polygen_length.push_back(raw_data.size()/2);
        }
        level_length.push_back(polygen_length.size());
    }

    /**
     * Convert polygen_length and level_length to json object.
    */
    Json::Value polygen_length_json=Json::arrayValue;
    for(int i=0;i<polygen_length.size();i++)
    {
        polygen_length_json[i]=polygen_length[i];
    }
    infor["polygon_length"]=polygen_length_json;

    Json::Value level_length_json=Json::arrayValue;
    for(int i=0;i<level_length.size();i++)
    {
        level_length_json[i]=level_length[i];
    }
    infor["levels_length"]=level_length_json;

    return;
}

void ccontour::print_json(Json::Value &c, const double xstart,const double xstep,const double ystart,const double ystep)
{
    Json::Value contourss=Json::arrayValue;

    for(int i=0;i<lines.size();i++)
    {
        Json::Value val;
        val["level"]=z[i];
        val["data"]=Json::arrayValue;
        val["spans"]=Json::arrayValue;
        for(int j=0;j<lines[i].size();j++)
        {   
            double xmin=10000.0;
            double xmax=-100000.0;
            double ymin=10000.0;
            double ymax=-100000.0;
            
            val["data"][j]=Json::arrayValue;
            for(int k=0;k<lines[i][j].x.size();k++)
            {
                val["data"][j][k]=Json::arrayValue;

                double xx=lines[i][j].x[k];
                double yy=lines[i][j].y[k];

                xx=xx*xstep+xstart;
                yy=yy*ystep+ystart;
                
                val["data"][j][k][0]=xx;
                val["data"][j][k][1]=yy;
                if(xx<xmin) xmin=xx;
                if(yy<ymin) ymin=yy;
                if(xx>xmax) xmax=xx;
                if(yy>ymax) ymax=yy;
            } 
            val["spans"][j]=Json::arrayValue;
            val["spans"][j][0]=xmin;
            val["spans"][j][1]=xmax;
            val["spans"][j][2]=ymin;
            val["spans"][j][3]=ymax;
        }
        contourss[i]=val;
    }

    c["contour"]=contourss;

    return;
};