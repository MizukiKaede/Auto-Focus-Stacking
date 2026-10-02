#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#define API extern "C" __declspec(dllexport)
API void probe_update(const uint8_t* classes, const uint8_t* rgb, uint8_t* lo, uint8_t* hi, uint16_t* count, int64_t n) {
    for (int64_t i=0;i<n;++i) {
        const int64_t p=int64_t(classes[i])*n+i;
        ++count[p];
        for(int c=0;c<3;++c) { lo[p*3+c]=std::min(lo[p*3+c],rgb[i*3+c]); hi[p*3+c]=std::max(hi[p*3+c],rgb[i*3+c]); }
    }
}
static float median(std::vector<float>& x) {
    const size_t k=x.size()/2;
    std::nth_element(x.begin(),x.begin()+k,x.end());
    const float upper=x[k];
    if(x.size()%2) return upper;
    const float lower=*std::max_element(x.begin(),x.begin()+k);
    return (lower+upper)*0.5f;
}
static float noise(std::vector<float>& x) {
    const float m=median(x);
    for(float& v:x) v=std::abs(v-m);
    const float mad=median(x);
    return float(std::max(double(std::numeric_limits<float>::epsilon()),double(m)+6.0*1.4826*double(mad)));
}
API void flat_noise(const uint8_t* bins,const float* vs,const float* gs,const float* fs,int64_t n,
                    float* ff,float* fv,float* fg,int64_t* samples,int64_t* fallbacks,
                    float* focus_floor,float* variance_floor,float* gradient_floor) {
    std::array<std::vector<int64_t>,16> groups;
    for(int64_t i=0;i<n;++i) groups[bins[i]].push_back(i);
    for(int b=0;b<16;++b) {
        ff[b]=fv[b]=fg[b]=0;
        const auto& indices=groups[b];
        if(indices.size()<64) continue;
        std::vector<float> values; values.reserve(indices.size());
        for(auto i:indices) values.push_back(vs[i]);
        const double pos=(values.size()-1)*0.35;
        const size_t k=size_t(pos);
        std::nth_element(values.begin(),values.begin()+k,values.end());
        const float a=values[k];
        const float upper=*std::min_element(values.begin()+k+1,values.end());
        const double fraction=pos-k;
        const float diff=upper-a;
        const double threshold=fraction>=0.5 ? double(upper)-double(diff)*(1.0-fraction) : double(a)+double(diff)*fraction;
        std::vector<float> v,g,f;
        for(auto i:indices) if(vs[i]<=threshold) {v.push_back(vs[i]);g.push_back(gs[i]);f.push_back(fs[i]);}
        if(v.size()<32) continue;
        samples[b]+=int64_t(v.size());
        ff[b]=noise(f);fv[b]=noise(v);fg[b]=noise(g);
    }
    bool observed[16];
    for(int b=0;b<16;++b) {
        focus_floor[b]=std::max(focus_floor[b],ff[b]);
        variance_floor[b]=std::max(variance_floor[b],fv[b]);
        gradient_floor[b]=std::max(gradient_floor[b],fg[b]);
        observed[b]=ff[b]>0;
    }
    for(int b=0;b<16;++b) if(!observed[b]) {
        bool found=false;
        for(int a:{b-1,b+1}) if(a>=0 && a<16 && observed[a]) {
            found=true; ff[b]=std::max(ff[b],ff[a]);fv[b]=std::max(fv[b],fv[a]);fg[b]=std::max(fg[b],fg[a]);
        }
        if(found) ++fallbacks[b];
    }
}
API void chroma_map(const uint8_t* rgb,uint8_t* chroma,int64_t n) {
    for(int64_t i=0;i<n;++i) {
        auto p=rgb+3*i;
        chroma[i]=std::max(p[0],std::max(p[1],p[2]))-std::min(p[0],std::min(p[1],p[2]));
    }
}

// Pixel loops retain NumPy float32 operation order; no fast-math or FMA.
API void material_classes(const uint8_t* hsv,const uint8_t* gray,uint8_t* out,int64_t n) {
    for(int64_t i=0;i<n;++i) {
        if(hsv[3*i+1]>=50) out[i]=1+((unsigned(hsv[3*i])+15)/30)%6+6*(gray[i]<64);
        else out[i]=13+(gray[i]>=64)+(gray[i]>=160);
    }
}
API void masked_colour(const float* rgb,const uint8_t* mask,float* out,int64_t n) {
    for(int64_t i=0;i<n;++i) for(int c=0;c<3;++c) out[3*i+c]=rgb[3*i+c]*float(mask[i]);
}
API void normalize_colour(float* field,const float* density,int64_t n) {
    for(int64_t i=0;i<n;++i) {
        const float d=std::max(density[i],1e-6f);
        for(int c=0;c<3;++c) field[3*i+c]=field[3*i+c]/d;
    }
}
static float clip01(float v) {return std::max(0.0f,std::min(1.0f,v));}
API void update_offset(const uint8_t* classes,uint8_t material,const float* delta,const float* density,
                       float* offset,uint8_t* confidence,int64_t n) {
    for(int64_t i=0;i<n;++i) if(classes[i]==material && density[i]>1e-6f) {
        const float magnitude=std::max(std::abs(delta[3*i]),std::max(std::abs(delta[3*i+1]),std::abs(delta[3*i+2])));
        float alpha=clip01((magnitude-1.0f)/3.0f);
        alpha=alpha*clip01((40.0f-magnitude)/20.0f);
        alpha=alpha*clip01(density[i]/0.2f);
        for(int c=0;c<3;++c) offset[3*i+c]=delta[3*i+c]*alpha;
        confidence[i]=1;
    }
}
API int64_t apply_colour(const uint8_t* rgb,const float* field,const uint8_t* valid,uint8_t* out,int64_t n) {
    int64_t changed=0;
    for(int64_t i=0;i<n;++i) {
        bool different=false;
        for(int c=0;c<3;++c) {
            const int64_t p=3*i+c;
            uint8_t value=rgb[p];
            if(valid[i]) {
                const float sum=float(rgb[p])+field[p];
                const float v=std::max(0.0f,std::min(255.0f,sum));
                int rounded=int(v);
                const float fraction=v-float(rounded);
                rounded+=(fraction>0.5f || (fraction==0.5f && (rounded&1)));
                value=uint8_t(rounded);
            }
            out[p]=value; different|=value!=rgb[p];
        }
        changed+=different;
    }
    return changed;
}
