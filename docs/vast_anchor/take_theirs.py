import sys
for path in sys.argv[1:]:
    L=open(path,encoding="utf-8",errors="replace").readlines()
    out=[];i=0;res=0
    while i<len(L):
        if L[i].startswith("<<<<<<<"):
            j=i+1
            while j<len(L) and not L[j].startswith("======="): j+=1
            k=j+1; theirs=[]
            while k<len(L) and not L[k].startswith(">>>>>>>"): theirs.append(L[k]); k+=1
            out.extend(theirs); res+=1; i=k+1
        else: out.append(L[i]); i+=1
    open(path,"w",encoding="utf-8").writelines(out)
    print(f"  {path}: took theirs for {res} conflict(s)")
