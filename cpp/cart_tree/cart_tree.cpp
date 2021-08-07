#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>


class Rect
{
    private:
        double x1, y1, x2, y2;

        double area;

        unsigned int land;

    public:
        std::vector<Rect *> below;

        Rect(double x1_, double y1_, double x2_, double y2_) : x1(x1_), y1(y1_), x2(x2_), y2(y2_) 
        {
            this->area = (x2 - x1)*(y2 - y1);
        }

        // We use the default constructor to construct root for the tree
        Rect()
        {
            // Define sea
            this->x1 = -std::numeric_limits<double>::max();
            this->y1 = -std::numeric_limits<double>::max();
            this->x2 =  std::numeric_limits<double>::max();
            this->y2 =  std::numeric_limits<double>::max();
            
            this->area = (x2 - x1)*(y2 - y1);


            land = 0;
        }
        
        double getArea() const
        {
            return this->area;
        }
 
        double getLand() const
        {
            return this->land;
        }
       
        void print() const
        {
            std::cout << this->x1 << " " << this->y1 << " "
                      << this->x2 << " " << this->y2 << std::endl;
        }

        bool contains(const Rect &rct) const
        {
            if ( (this->x2 > rct.x2) && (this->y2 > rct.y2) && 
                 (this->x1 < rct.x1) && (this->x1 < rct.x1) 
               )
                return true;
            else
                return false;
        }


        void insert(Rect *rct)
        {
            if (this->below.empty())
            {
                this->below.push_back( rct );
                return;
            }
            Rect *tmp = this;

            unsigned int ctr = 0;

            std::queue<Rect *> traverse; 
            traverse.push(this);
            while( !traverse.empty() )
            {
                auto loc = traverse.front();
                traverse.pop();

                for(auto it: loc->below )
                {
                    if ( it->contains(*rct) )
                    {
                        tmp = it;
                        traverse.push(it);
                        ctr++;
                    }
                }
            }

            tmp->below.push_back( rct );

            if (ctr%2 == 0) land++;

            return;
        }

};



int main()
{
    std::ifstream fl("inp.txt");

    int n;
    fl >> n;

    std::vector<Rect> rectVec;
    for(int i=0; i < n; i++)
    {
        double x1, y1, x2, y2;

        fl >> x1 >> y1 >> x2 >> y2;

        rectVec.push_back(Rect(x1, y1, x2, y2));
    }

    std::sort( rectVec.begin(), rectVec.end(), [](const Rect &x, const Rect &y){ return x.getArea() > y.getArea(); } );
    
    auto hrTree = Rect();

    for(int i = 0; i < rectVec.size(); i++)
    {
        Rect *pt = new Rect(rectVec[i]);
        hrTree.insert(pt);
    }

    std::cout << "Number of land regions are " << hrTree.getLand() << std::endl;


    return 0;
}























