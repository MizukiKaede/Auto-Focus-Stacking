#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#define API extern "C" __declspec(dllexport)
using U8 = uint8_t;
using U16 = uint16_t;
using I64 = int64_t;

API int fast_core_abi() { return 3; }
static U8 byte_round(float v) {
    return U8(std::min(255.0f, std::max(0.0f, std::nearbyint(v))));
}
API void rgb_chroma(const U8* rgb, U8* out, int h, int w, I64 stride) {
    for (int y=0;y<h;++y) for(int x=0;x<w;++x) {
        const U8* p=rgb+y*stride+x*3;
        out[I64(y)*w+x]=std::max({p[0],p[1],p[2]})-std::min({p[0],p[1],p[2]});
    }
}
API void independent_detail(const float* detail,const float* broad,const float* gradient,
                            float floor,U8* out,I64 n) {
    for(I64 i=0;i<n;++i)
        out[i]=detail[i]>floor && (detail[i]>2.0f*broad[i] || detail[i]>16.0f*gradient[i]);
}
API void focus_fields(const float* xx,const float* yy,const float* xy,const float* gradient,
                      const float* variance,const float* detail,float floor,
                      float* base,float* sharp,U8* texture,I64 n) {
    for(I64 i=0;i<n;++i) {
        const float trace=xx[i]+yy[i], d=xx[i]-yy[i];
        base[i]=0.65f*gradient[i]+0.35f*trace+0.10f*std::max(variance[i],0.0f);
        const float minor=0.5f*(trace-std::sqrt(std::max(d*d+4.0f*xy[i]*xy[i],0.0f)));
        texture[i]=minor>0.12f*trace && detail[i]>floor && trace>2e-5f;
        sharp[i]=(gradient[i]>1e-4f && detail[i]>floor)
                 ? gradient[i]*std::sqrt(std::max(detail[i],0.0f)) : 0.0f;
    }
}
API void proxy_winners(const U8* valid,const U8* evidence,const U8* texture,
                       const float* score,const float* detail,U16 index,
                       float* best,U16* labels,float* detail_best,U16* detail_owner,
                       float* local_best,U16* local_owner,U8* textured,U8* local_better,I64 n) {
    for(I64 i=0;i<n;++i) {
        local_better[i]=0;
        if(!valid[i]) continue;
        if(score[i]>best[i]) { best[i]=score[i]; labels[i]=index; }
        if(detail[i]>detail_best[i]) {detail_best[i]=detail[i];detail_owner[i]=index;}
        if(evidence[i] && detail[i]>local_best[i]) {
            local_best[i]=detail[i];local_owner[i]=index;local_better[i]=1;
        }
        textured[i] |= texture[i];
    }
}
API void preserve_detail(U16* result,const U16* propagated,const float* propagated_detail,
                         const float* local_best,const U16* local_owner,const U8* active,
                         U8* independent,I64 n) {
    for(I64 i=0;i<n;++i) {
        independent[i]=active[i] && local_best[i]>0.0f && propagated_detail[i]<0.7f*local_best[i];
        if(active[i]) result[i]=independent[i]?local_owner[i]:propagated[i];
    }
}
API void neutral_seed_filter(const I64* positions,const float* details,const float* local_best,
                             const U8* chroma,const U8* foreground,const U8* material,U8* use,I64 n) {
    for(I64 j=0;j<n;++j) {
        const I64 i=positions[j];
        use[j]=local_best[i]>0.0f && chroma[i]<65 && foreground[i] && material[i]
               && details[j]>=0.7f*local_best[i];
    }
}
API void neutral_update(const U8* targets,const U8* material,const float* strength,
                         float* best,U16* owner,U8* chroma,U16 index,I64 n) {
    for(I64 i=0;i<n;++i) if(targets[i] && material[i] && strength[i]>best[i]) {
        best[i]=strength[i];owner[i]=index;chroma[i]=0;
    }
}
API void neutral_apply(U16* result,const U16* owner,const float* best,const U8* chroma,
                        const U8* protected_pixels,I64* counts,I64 n) {
    I64 guarded=0,changed=0;
    for(I64 i=0;i<n;++i) if(best[i]>1e-6f && chroma[i]<65 && !protected_pixels[i]) {
        ++guarded;changed+=result[i]!=owner[i];result[i]=owner[i];
    }
    counts[0]=guarded;counts[1]=changed;
}
API void support_inputs(const float* strength,const U8* edge,float* present,float* masked,I64 n) {
    for(I64 i=0;i<n;++i) {present[i]=edge[i]?1.0f:0.0f;masked[i]=strength[i]*present[i];}
}
API int nearest_support(const U8* present,const float* density,const float* averaged,
                        const float* distance,const int32_t* nearest,float radius,float* out,I64 n) {
    try {
        int32_t maximum=0;
        for(I64 i=0;i<n;++i) maximum=std::max(maximum,nearest[i]);
        std::vector<float> values(size_t(maximum)+1,0.0f);
        for(I64 i=0;i<n;++i) if(present[i])
            values[nearest[i]]=averaged[i]/std::max(density[i],1e-6f);
        const float denominator=std::max(2.0f,radius/4.0f);
        for(I64 i=0;i<n;++i) {
            const float weight=std::min(1.0f,std::max(0.0f,(radius+1.0f-distance[i])/denominator));
            out[i]=values[nearest[i]]*weight;
        }
        return 0;
    } catch(...) { return -1; }
}
API void contour_energy(const float* gradient,const float* detail,const U8* seeds,
                         float* present,float* weighted,I64 n) {
    for(I64 i=0;i<n;++i) {
        present[i]=seeds[i]?1.0f:0.0f;
        weighted[i]=gradient[i]*std::sqrt(std::max(detail[i],0.0f))*present[i];
    }
}
API void normalize_field(float* field,const float* density,I64 n) {
    for(I64 i=0;i<n;++i) field[i]/=std::max(density[i],1e-6f);
}
API void native_rank(float* best,U16* owner,float* reference_score,const U8* mask,
                      const U8* full_valid,const float* score,U8* better,
                      int h,int w,int x,int y,int full_w,U16 index,int reference) {
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
        const I64 i=I64(r)*w+c, f=I64(y+r)*full_w+x+c;
        const bool usable=mask[i] && full_valid[f];
        better[i]=usable && score[i]>best[i];
        if(better[i]) {best[i]=score[i];owner[i]=index;}
        if(reference && usable) reference_score[i]=score[i];
    }
}
API void native_capture(U8* rgb,U8* reference_rgb,const U8* full_rgb,const U8* better,
                         int h,int w,int x,int y,int full_w,int reference) {
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
        const I64 i=I64(r)*w+c, f=I64(y+r)*full_w+x+c;
        if(better[i]) std::memcpy(rgb+i*3,full_rgb+f*3,3);
        if(reference) std::memcpy(reference_rgb+i*3,full_rgb+f*3,3);
    }
}
API void native_select(U8* rgb,const U8* reference_rgb,U16* owner,const U16* labels,
                       const float* best,const float* reference_score,const U8* mask,
                       U8* covered,int h,int w,int x,int y,int full_w,U16 reference,I64* counts) {
    I64 references=0,replaced=0;
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
        const I64 i=I64(r)*w+c, f=I64(y+r)*full_w+x+c;
        const bool use=mask[i] && best[i]>0.0f;
        if(use && reference_score[i]>=0.9f*best[i]) {
            std::memcpy(rgb+i*3,reference_rgb+i*3,3);owner[i]=reference;++references;
        }
        if(use && owner[i]!=labels[i]) ++replaced;
        covered[f]=use;
    }
    counts[0]=references;counts[1]=replaced;
}
API void native_blend(U8* full_rgb,const U8* rgb,const U8* covered,const float* distance,
                      float fade,int h,int w,int x,int y,int full_w) {
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
        const I64 i=I64(r)*w+c, f=I64(y+r)*full_w+x+c;
        if(!covered[f]) continue;
        const float t=std::min(distance[f]/fade,1.0f), a=t*t*(3.0f-2.0f*t);
        for(int k=0;k<3;++k) full_rgb[f*3+k]=byte_round(full_rgb[f*3+k]*(1.0f-a)+rgb[i*3+k]*a);
    }
}

struct Render {
    std::vector<U16> labels;
    std::vector<U8> band,copied;
    std::vector<I64> seam;
    std::vector<float> colors,weights;
    I64 n;
    Render(const U16* l,const U8* b,I64 count): labels(l,l+count),band(b,b+count),copied(count,0),n(count) {
        for(I64 i=0;i<n;++i) if(band[i]) seam.push_back(i);
        colors.resize(seam.size()*3,0.0f);weights.resize(seam.size(),0.0f);
    }
};
API void* render_create(const U16* labels,const U8* band,I64 n) {
    try {return new Render(labels,band,n);} catch(...) {return nullptr;}
}
API void render_destroy(void* p) {delete static_cast<Render*>(p);}
API I64 render_seams(void* p) {return I64(static_cast<Render*>(p)->seam.size());}
API I64 render_owner(void* p,U16 index,U8* owner) {
    auto& s=*static_cast<Render*>(p);I64 count=0;
    for(I64 i=0;i<s.n;++i) {owner[i]=s.labels[i]==index;count+=owner[i];}
    return count;
}
API void render_add(void* p,U16 index,const U8* rgb,const U8* valid,
                     const float* feather,U8* output) {
    auto& s=*static_cast<Render*>(p);
    for(I64 i=0;i<s.n;++i) if(s.labels[i]==index && !s.band[i] && valid[i]) {
        std::memcpy(output+i*3,rgb+i*3,3);s.copied[i]=1;
    }
    if(!feather) return;
    for(size_t j=0;j<s.seam.size();++j) {
        const I64 i=s.seam[j];const float weight=feather[i]*float(valid[i]);
        for(int c=0;c<3;++c) s.colors[j*3+c]+=float(rgb[i*3+c])*weight;
        s.weights[j]+=weight;
    }
}
API I64 render_finish(void* p,U8* output) {
    auto& s=*static_cast<Render*>(p);
    for(size_t j=0;j<s.seam.size();++j) if(s.weights[j]>1e-8f) {
        const I64 i=s.seam[j];
        for(int c=0;c<3;++c) output[i*3+c]=byte_round(s.colors[j*3+c]/s.weights[j]);
        s.copied[i]=1;
    }
    I64 missing=0;for(auto v:s.copied) missing+=!v;return missing;
}
