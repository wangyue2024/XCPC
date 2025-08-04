#include<bits/stdc++.h>
using namespace std;
#define D long double
#define P pair<D,D>
#define x first
#define y second 
const long double eps=1e-15;
const long double pai=acos((long double)-1);
#define sqrt(x) sqrt(max((x),(D)0))
struct Line
{
    P a,v;
};

P operator-(P a,P b){
    return {a.x-b.x,a.y-b.y};
}
P operator+(P a,P b){
    return {a.x+b.x,a.y+b.y};
}
P operator*(D v,P a){
    return {v*a.x,v*a.y};
}
int sign(long double x){
    if(fabs(x)<eps)return 0;
    if(x<0)return -1;
    return 1;
}
int sign(long long x){
    if(x==0)return 0;
    if(x<0)return -1;
    return 1;
}
D dot(P a,P b){
    return a.x*b.x+a.y*b.y;
}
P project(P d,Line b){
    return b.a+(dot(d-b.a,b.v)/dot(b.v,b.v))*b.v;
}
P reflect(P d,Line b){
    return 2*project(d,b)-d;
}
D cross(P a,P b){
    return a.x*b.y-a.y*b.x;
}
D dist2(P a){
    return dot(a,a);
}
D dist(P a){
    return sqrt(dot(a,a));
}

bool if_intersection(P a,P b,P c,P d){
    if(max(a.x,b.x)<min(c.x,d.x)||max(a.y,b.y)<min(c.y,d.y)||max(c.x,d.x)<min(a.x,b.x)||max(c.y,d.y)<min(a.y,b.y))return 0;
    return cross(b-a,c-a)*cross(b-a,d-a)<=0&&cross(d-c,a-c)*cross(d-c,b-c)<=0;
}
P intersection(P a,P b,P c,P d){
    P v=a-c;
    D t=cross(d-c,a-c)/cross(b-a,d-c);
    return a+t*(b-a);
}

D dist_point_line(P a,P b,P c){
    return fabs(cross(a-b,c-b)/dist(c-b));
}
D dist_point_seg(P a,P b,P c){
    if(dot(a-b,c-b)<=0)return dist(a-b);
    if(dot(a-c,b-c)<=0)return dist(a-c);
    return dist_point_line(a,b,c);
}
D dist_seg_seg(P a,P b,P c,P d){
    #define f  dist_point_seg
    if(if_intersection(a,b,c,d))return 0;
    return min({f(a,c,d),f(b,c,d),f(c,a,b),f(d,a,b)});
    #undef f
}

D area(int n,P d[]){
    D ans=0;
    for(int i=1;i<n;++i){
        ans+=cross(d[i]-d[1],d[i+1]-d[1]);
    }
    return ans/2;
}
bool on_seg(P a,P b,P c){
    return sign(cross(b-a,c-a))==0&&sign(dot(b-a,c-a))<=0;
}

D arg(P a,P b){
    // 不能使用asin，可以用acos再判断方向
    // this can lead to WA RE because the value can be 1.0000000000000001
    // D t=acos(dot(a,b)/dist(a)/dist(b));
    // if(sign(cross(a,b))<0)t*=-1;
    // return t;

    // 或者用atan2
    return atan2(cross(a,b),dot(a,b));
}
D area(P a,P b,P c){
    return fabs(cross(b-a,c-a));
}
bool left(P d,Line l){
    return sign(cross(l.v,d-l.a))>=0;
}
int contain(int n,P a[],P p){
    // 2 : contain
    // 1 : on edge
    a[n+1]=a[1];
    D ans=0;
    for(int i=1;i<=n;++i){
        if(on_seg(p,a[i],a[i+1]))return 1;
        ans+=arg(a[i]-p,a[i+1]-p);
    }
    if(sign(ans)==0)return 0;
    return 2;
}
int contain2(int n,P a[],P p){
    // the second opti, solve the inter problem
    int op=0;
    a[n+1]=a[1];
    for(int i=1;i<=n;++i){
        if(on_seg(p,a[i],a[i+1]))return 1;
        P p1=a[i]-p,p2=a[i+1]-p;
        if(p1.y>p2.y)swap(p1,p2);
        if(p1.y<=0&&p2.y>0&&cross(p1,p2)>0)op^=1;
    }
    return op?2:0;
}

const int N =1.1e6;
int sk[N];
int hull(int n,P a[],P ans[]){
    int top,bas;
    bas=1;top=0;
    sort(a+1,a+1+n);
    for(int i=1;i<=n;++i){
        while(top>bas&&cross(a[i]-a[sk[top-1]],a[sk[top]]-a[sk[top-1]])>0)top--;
        sk[++top]=i;
    }
    bas=top;
    for(int i=n-1;i;--i){
        while(top>bas&&cross(a[i]-a[sk[top-1]],a[sk[top]]-a[sk[top-1]])>0)top--;
        sk[++top]=i;
    }
    top--;
    for(int i=1;i<=top;++i){
        ans[i]=a[sk[i]];
    }
    return top;
}

D max_dist(int n,P a[]){
    //  seg_based
    D ans=0;
    for(int i=1;i<=n;++i)a[i+n]=a[i];
    for(int i=1,j=3;i<=n;++i){
        while(area(a[j+1],a[i],a[i+1])>=area(a[j],a[i],a[i+1]))j++;
        // > is incorrect 
        ans=max(ans,max(dist2(a[j]-a[i]),dist2(a[j]-a[i+1])));
    }
    return ans;
}
D cut_hull(int n,P a[],Line l,P ans[]){
    int ip=0;
    a[n+1]=a[1];
    for(int i=1;i<=n;++i){
        if(left(a[i],l))ans[++ip]=a[i];
        if(sign(cross(l.v,a[i]-l.a)*cross(l.v,a[i+1]-l.a))<0)ans[++ip]=intersection(a[i],a[i+1],l.a,l.a+l.v);
    }
    return area(ip,ans);
}
void pr(P d){
    printf("%.10Lf %.10Lf\n",d.x,d.y);
}

P a[N],ans[N];
//divide and conquer
bool cmpy(P a,P b){return a.y<b.y;}
D solve(int l,int r){
    if(l==r)return 1e18;
    if(l+1==r)return dist(a[l]-a[r]);
    D d=1e18;
    int mid=l+r>>1;
    d=min(solve(l,mid),solve(mid+1,r));
    int ip=0;
    for(int i=l;i<=r;++i){
        if(fabs(a[i].x-a[mid].x)<=d)ans[++ip]=a[i];
    }
    sort(ans+1,ans+1+ip,cmpy);
    for(int i=1;i<=ip;++i){
        for(int j=i+1;j<=ip&&ans[j].y-ans[i].y<=d;++j)d=min(d,dist(ans[j]-ans[i]));
    }
    return d;
}
P norm(P a){
    return {a.y,-a.x};
}
void g(P &a){
    cin>>a.x>>a.y;
}
P mid_angle(P a,P b){
    a=1/dist(a)*a;
    b=1/dist(b)*b;
    return a+b;
}
P interheart(P a,P b,P c){
    return intersection(a,mid_angle(b-a,c-a)+a,b,mid_angle(a-b,c-b)+b);
}
P outerheart(P a,P b,P c){
    return intersection(0.5*(a+b),0.5*(a+b)+norm(a-b),0.5*(b+c),0.5*(b+c)+norm(b-c));
}
void clintersection(P d,D r,P a,P b,P &ans1,P&ans2){
    D dis=dist_point_line(d,a,b);
    D res=sqrt(max((long double)0.0,r*r-dis*dis));
    P mid=project(d,{a,b-a}),v=1/dist(b-a)*(b-a);
    ans1=mid+res*v;
    ans2=mid-res*v;
}

void ccintersection(P s1,D r1,P s2,D r2,P &ans1,P&ans2){
    D R=dist(s1-s2);
    #define l(x) ((x)*(x))
    P a=0.5*(s1+s2)+(l(r1)-l(r2))/(2*l(R))*(s2-s1),b=0.5*sqrt(2*(l(r1)+l(r2))/l(R) -l(l(r1)-l(r2))/l(l(R))-1)*(P){s2.y-s1.y,s1.x-s2.x};
    ans1=a+b;ans2=a-b;

}
D areashan(P d,D r,P a,P b){
    return fabs(arg(b-d,a-d))*r*r/2-fabs(cross(a-d,b-d)/2);
}
D areaccintersection(P s1,D r1,P s2,D r2){
    D d=dist(s1-s2);
    if(r1<r2)swap(s1,s2),swap(r1,r2);
    if(sign(d-r1-r2)>=0)return 0;
    if(sign(fabs(r1-r2)-d)>=0)return pai*r2*r2;
    P ans1,ans2;
    ccintersection(s1,r1,s2,r2,ans1,ans2);
    D a1=areashan(s1,r1,ans1,ans2),a2=areashan(s2,r2,ans1,ans2);
    if(cross(ans1-ans2,s1-ans2)*cross(ans1-ans2,s2-ans2)>0)return pai*r2*r2-a2+a1;
    return a1+a2;
}
D areacpintersection(D r,P a[],int n){
    P b={0,0};
    D ans=0;
    a[n+1]=a[1];
    for(int i=1;i<=n;++i){
        if(dist(a[i])<=r&&dist(a[i+1])<=r)ans+=cross(a[i],a[i+1])/2;
        else if(dist(a[i])>r&&dist(a[i+1])>r){
            ans+=arg(a[i],a[i+1])*r*r/2;
            if(on_seg(project(b,{a[i],a[i+1]-a[i]}),a[i],a[i+1])){
                P ans1,ans2;
                clintersection(b,r,a[i],a[i+1],ans1,ans2);
                if((a[i]<=ans1&&ans1<=ans2)||(a[i]>=ans1&&ans1>=ans2));
                else swap(ans1,ans2);
                ans-=arg(ans1,ans2)*r*r/2-cross(ans1,ans2)/2;
            }
        }
        else{
            P ans1,ans2;
            clintersection(b,r,a[i],a[i+1],ans1,ans2);
            if(!on_seg(ans1,a[i],a[i+1]))ans1=ans2;
            if(dist(a[i])<r)ans+=cross(a[i],ans1)/2+arg(ans1,a[i+1])*r*r/2;
            else ans+=arg(a[i],ans1)*r*r/2+cross(ans1,a[i+1])/2;
        }
    }
    return ans;
}
/**
 * @brief 计算两个圆的交点
 * @param c1 第一个圆 {圆心, 半径}
 * @param c2 第二个圆 {圆心, 半径}
 * @return 返回一个 vector<P>，包含 0, 1, 或 2 个交点。
 *         如果两圆重合，返回空vector。
 */
vector<P> get_circle_intersections(pair<P, D> c1_in, pair<P, D> c2_in) {
    P p1 = c1_in.first;
    D r1 = c1_in.second;
    P p2 = c2_in.first;
    D r2 = c2_in.second;

    vector<P> res;
    P v = p2 - p1;
    D d2 = dist2(v);
    D d = sqrt(d2);

    // Case 1: 同心圆
    if (sign(d) == 0) {
        // 如果半径也相同，则重合，有无限交点，按无交点处理
        return res;
    }

    // Case 2: 相离或内含
    if (sign(d - (r1 + r2)) > 0 || sign(d - abs(r1 - r2)) < 0) {
        return res;
    }

    // 计算 P1 到 P1P2 与公共弦交点 M 的距离 a
    D a = (r1 * r1 - r2 * r2 + d2) / (2.0 * d);

    // 计算公共弦半长 h
    D h2 = r1 * r1 - a * a;
    D h = sqrt(h2); // 使用模板中定义的安全sqrt

    // 计算垂足 M 的坐标
    P M = p1 + (a / d) * v;

    // 计算与向量 v 垂直的向量 v_perp
    // v = (v.x, v.y), v_perp = (-v.y, v.x)
    P v_perp = {-v.y, v.x};

    // 计算两个交点
    res.push_back(M + (h / d) * v_perp);
    if (sign(h) != 0) { // 如果不是切点 (h > 0)，才有第二个交点
        res.push_back(M - (h / d) * v_perp);
    }

    return res;
}
namespace min_cir_cover
{
    struct C
    {
        D r;
        P d;
    };
    C cir(P a,P b){
        return (C){dist(b-a)/2,0.5*(a+b)};
    }
    C cir(P a,P b,P c){
        P d=intersection(0.5*(a+b),0.5*(a+b)+norm(a-b),0.5*(b+c),0.5*(b+c)+norm(b-c));
        return (C){dist(d-a),d};
    }
    C find(int n,P d[]){
        random_shuffle(d+1,d+1+n);
        C x=cir(d[1],d[2]);
        for(int i=3;i<=n;++i){
            if(dist(d[i]-x.d)>x.r+eps){
                x=cir(d[i],d[1]);
                for(int j=2;j<i;++j){
                    if(dist(d[j]-x.d)>x.r+eps){
                        x=cir(d[i],d[j]);
                        for(int k=1;k<j;++k){
                            if(dist(d[k]-x.d)>x.r+eps){
                                x=cir(d[i],d[j],d[k]);
                            }
                        }
                    }
                }
            }
        }
        return x;
    }
};

signed main(){
    int n;
    cin>>n;
    for(int i=1;i<=n;++i)cin>>a[i].x>>a[i].y;
    auto x=min_cir_cover::find(n,a);
    printf("%.20Lf\n%.20Lf %.20Lf",x.r,x.d.x,x.d.y);

}