#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "randgen.h"

extern double M;
extern int PREC;

double kiss(unsigned long *seed)
/* Generator proposed by Marsaglia and Zaman, 1993. See
   Robert and Casella (1999, pages 41-43) for details.  
   Watch out: the last line
        x = ((double) (*i+*j+*k)*exp(-32*log(2.0)));
   must be calibrated, depending on the precision 
   of the computer. */
{
  seed[1] = seed[1] ^ (seed[1]<<17);
  seed[2] = (seed[2] ^ (seed[2]<<18)) & 0x7FFFFFFF;
  seed[0] = 69069*seed[0]+23606797;
  seed[1] ^= (seed[1]>>15);
  seed[2] ^= (seed[2]>>13);
  return((seed[0]+seed[1]+seed[2])*M);
}

int runiform_n(int n,unsigned long *seed)
{
  int i;

  i = (int)floor(n*kiss(seed));
  if (i == n)
    return(i-1);
  else 
    return(i);
}

unsigned long int runiform_long_n(unsigned long n,unsigned long *seed)
{
  unsigned long int i;

  i = (unsigned long int)floorl(n*kiss(seed));
  if (i == n)
    return(i-1);
  else 
    return(i);
}

void permute_sample(double *v,int len,unsigned long *seed) 
{
  int i,j;
  double x;
  
  for (i=len;i>0;i--) {
    j = runiform_n(i,seed);
    x = v[j];
    v[j] = v[i-1];
    v[i-1] = x;
  }
}

void permute_sample_int(int *v,int len,unsigned long *seed) 
{
  int i,j;
  int x;
  
  for (i=len;i>0;i--) {
    j = runiform_n(i,seed);
    x = v[j];
    v[j] = v[i-1];
    v[i-1] = x;
  }
}

long rmultinomial(double *prob,long len,unsigned long *seed)
{
 long i;
 double y;

 y = kiss(seed);
 for (i=0;i<len;i++)
   if (y < prob[i])
     return i;
 return len-1;
}


double runif_atob(unsigned long *seed,double a,double b)
{
 return ((b-a)*kiss(seed) + a);
}


double rexp(double beta,unsigned long *seed)
{
 return -log(kiss(seed))/beta;
}

double gasdev(unsigned long *seed)
//double snorm(unsigned long *seed)
{
	static int iset=0;
	static double gset;
	double fac,rsq,v1,v2;
	
	if  (iset == 0) {
		do {
			v1=2.0*kiss(seed)-1.0;
			v2=2.0*kiss(seed)-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}

double sgamma(double a,unsigned long* seed)
/*
 **********************************************************************
 
 
 (STANDARD-)  G A M M A  DISTRIBUTION
 
 
 **********************************************************************
 **********************************************************************
 
 PARAMETER  A >= 1.0  !
 
 **********************************************************************
 
 FOR DETAILS SEE:
 
 AHRENS, J.H. AND DIETER, U.
 GENERATING GAMMA VARIATES BY A
 MODIFIED REJECTION TECHNIQUE.
 COMM. ACM, 25,1 (JAN. 1982), 47 - 54.
 
 STEP NUMBERS CORRESPOND TO ALGORITHM 'GD' IN THE ABOVE PAPER
 (STRAIGHTFORWARD IMPLEMENTATION)
 
 Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of
 SUNIF.  The argument IR thus goes away.
 
 **********************************************************************
 
 PARAMETER  0.0 < A < 1.0  !
 
 **********************************************************************
 
 FOR DETAILS SEE:
 
 AHRENS, J.H. AND DIETER, U.
 COMPUTER METHODS FOR SAMPLING FROM GAMMA,
 BETA, POISSON AND BINOMIAL DISTRIBUTIONS.
 COMPUTING, 12 (1974), 223 - 246.
 
 (ADAPTED IMPLEMENTATION OF ALGORITHM 'GS' IN THE ABOVE PAPER)
 
 **********************************************************************
 INPUT: A =PARAMETER (MEAN) OF THE STANDARD GAMMA DISTRIBUTION
 OUTPUT: SGAMMA = SAMPLE FROM THE GAMMA-(A)-DISTRIBUTION
 COEFFICIENTS Q(K) - FOR Q0 = SUM(Q(K)*A**(-K))
 COEFFICIENTS A(K) - FOR Q = Q0+(T*T/2)*SUM(A(K)*V**K)
 COEFFICIENTS E(K) - FOR EXP(Q)-1 = SUM(E(K)*Q**K)
 PREVIOUS A PRE-SET TO ZERO - AA IS A', AAA IS A"
 SQRT32 IS THE SQUAREROOT OF 32 = 5.656854249492380
 */
{
	extern double fsign( double num, double sign );
/*	static double q1 = 0.04166669;
	static double q2 = 0.02083148;
	static double q3 = 0.00801191;
	static double q4 = 0.00144121;
	static double q5 = -0.00007388;
	static double q6 = 0.00024511;
	static double q7 = 0.00024240;
	static double a1 = 0.3333333;
	static double a2 = -0.2500030;
	static double a3 = 0.2000062;
	static double a4 = -0.1662921;
	static double a5 = 0.1423657;
	static double a6 = -0.1367177;
	static double a7 = 0.1233795;
	static double e1 = 1.0000000;
	static double e2 = 0.4999897;
	static double e3 = 0.1668290;
	static double e4 = 0.0407753;
	static double e5 = 0.0102930;*/
	
	
	static double q1 = 0.0416666664;
	static double q2 = 0.0208333723;
	static double q3 = 0.0079849875;
	static double q4 = 0.0015746717;
	static double q5 = -0.0003349403;
	static double q6 = 0.0003340332;
	static double q7 = 0.0006053049;
	static double q8 = -0.0004701849;
	static double q9 = 0.0001710320;
	static double a1 = 0.333333333;
	static double a2 = -0.249999949;
	static double a3 = 0.199999867;
	static double a4 = -0.166677482;
	static double a5 = 0.142873973;
	static double a6 = -0.124385581;
	static double a7 = 0.110368310;
	static double a8 = 0.112750886;
	static double a9 = 0.104089866;
	static double e1 = 1.000000000;
	static double e2 = 0.499999994;
	static double e3 = 0.166666848;
	static double e4 = 0.041664508;
	static double e5 = 0.008345522;
	static double e6 = 0.001353826;
	static double e7 = 0.000247453;
	static double aa = 0.0;
	static double aaa = 0.0;
	static double sqrt32 = 5.65685424949238;
	static double sgamma,s2,s,d,t,x,u,r,q0,b,si,c,v,q,e,w,p;
	
    if(a == aa) 
		goto S2;
    if(a < 1.0) 
		goto S13;

S1: // STEP  1:  RECALCULATIONS OF S2,S,D IF A HAS CHANGED
    aa = a;
    s2 = a-0.5;
    s = sqrt(s2);
    d = sqrt32-12.0*s;
	
S2: // STEP  2:  T=STANDARD NORMAL DEVIATE,	 X=(S,1/2)-NORMAL DEVIATE.	 IMMEDIATE ACCEPTANCE (I)
    t = snorm(seed);
    x = s+0.5*t;
    sgamma = x*x;
    if(t >= 0.0) 
		return sgamma;

S3: // STEP  3:  U= 0,1 -UNIFORM SAMPLE. SQUEEZE ACCEPTANCE (S)
    u = kiss(seed);
    if(d*u <= t*t*t) 
		return sgamma;

S4: // STEP  4:  RECALCULATIONS OF Q0,B,SI,C IF NECESSARY
	if (a != aaa) {
		aaa = a;
		r = 1.0/ a;
		q0 = r*(q1+r*(q2+r*(q3+r*(q4+r*(q5+r*(q6+r*(q7+r*(q8+r*q9))))))));
		if (a <= 3.686) {
			b = 0.463+s+0.178*s2;
			si = 1.235;
			c = 0.195/s-0.079+0.16*s;
		}
		else if (a <= 13.022) {
			b = 1.654+0.0076*s2;
			si = 1.68/s+0.275;
			c = .062/s+0.024;
		}
		else {
			b = 1.77;
			si = 0.75;
			c = 0.1515/s;			
		}
	}

S5: //  NO QUOTIENT TEST IF X NOT POSITIVE
    if(x <= 0.0) 
		goto S8;
	
S6: // CALCULATION OF V AND QUOTIENT Q
    v = t/(s+s);
    if(fabs(v) > 0.25) 
		q = q0-s*t+0.25*t*t+(s2+s2)*log1p(v);
	else
		q = q0+0.5*t*t*v*(a1+v*(a2+v*(a3+v*(a4+v*(a5+v*(a6+v*(a7+v*(a8+v*a9))))))));

S7: //  QUOTIENT ACCEPTANCE (Q)
	if(log1p(-u) <= q) return sgamma;

S8: // E=STANDARD EXPONENTIAL DEVIATE  U= 0,1 -UNIFORM DEVIATE 	 T=(B,SI)-DOUBLE EXPONENTIAL (LAPLACE) SAMPLE
    e = rexp(1.0,seed);
    u = kiss(seed);
    u += (u-1.0);
    t = b+fsign(si*e,u);

S9: //   REJECTION IF T .LT. TAU(1) = -.71874483771719
	if(t <= -0.71874483771719) 
		goto S8;

S10: // CALCULATION OF V AND QUOTIENT Q
	v = t/(s+s);
	if(fabs(v) > 0.25) 
		q = q0-s*t+0.25*t*t+(s2+s2)*log1p(v);
	else
		q = q0+0.5*t*t*v*(a1+v*(a2+v*(a3+v*(a4+v*(a5+v*(a6+v*(a7+v*(a8+v*a9))))))));

S11: // HAT ACCEPTANCE (H) (IF Q NOT POSITIVE GO TO STEP 8)
	if (q <= 0.5)
		w = q*(e1+q*(e2+q*(e3+q*(e4+q*(e5+q*(e6+q*e7))))));
	else 
		w = exp(q)-1.0;
   if(q <= 0.0 || c*fabs(u) > w*exp(e-0.5*t*t)) 
		goto S8;
S12: 
    x = s+0.5*t;
    sgamma = x*x;
    return sgamma;
	
S13: // ALTERNATE METHOD FOR PARAMETERS A BELOW 1  (.3678794=EXP(-1.))
    aa = 0.0;
    b = 1.0+0.3678794*a;

S14:
    p = b*kiss(seed);
    if(p >= 1.0) 
		goto S15;
    sgamma = exp(log(p)/ a);
    if(rexp(1.0,seed) < sgamma) 
		goto S14;
    return sgamma;

S15:
    sgamma = -log((b-p)/ a);
    if(rexp(1.0,seed) < (1.0-a)*log(sgamma)) 
		goto S14;
    return sgamma;
}

double fsign( double num, double sign )
/* Transfers sign of argument sign to argument num */
{
	if ( ( sign>0.0 && num<0.0 ) || ( sign<0.0 && num>0.0 ) )
		return -num;
	else return num;
}

double compute_snorm(double *a,double *d,double *t,double *h,const int bits,const int log2bits,unsigned long *seed)
{
	int i;
	double snorm,u,s,ustar,aa,w,y,tt;
	
S1: u = 0;
    while (u <= 1e-37) 
		u = kiss(seed);
    s = 0.0;
    if(u > 0.5) s = 1.0;
    u += (u-s);
S2:
    u = bits*u;
    i = (int)floor(u);
	//    if (i>bits-1) i=bis-1;
    if(i == 0) 
		goto S9;
S3: // start center
    ustar = u-(double)i;
    aa = *(a+i);
S4:
    if(ustar > *(t+i)) {
		w = (ustar-*(t+i))**(h+i);
		goto S17;
	}
S5: 	
    u = kiss(seed);
    w = u*(*(a+i+1)-aa);
    tt = (0.5*w+aa)*w;
S6:
	if (ustar > tt)
		goto S17;
S7:	
	u = kiss(seed);
	if (ustar < u) {
		ustar = kiss(seed);
		goto S4;
	}
S8:	
	tt = u;
    ustar = kiss(seed);
	goto S6;
S9:	// start tail
    i = log2bits+1;
    aa = *(a+bits);
S10:
	u += u;
	if (u >= 1)
		goto S12;
S11:
    aa += *(d+i);
    i += 1;
	goto S10;
S12:
    u -= 1.0;
S13:
    w = u**(d+i);
    tt = (0.5*w+aa)*w;
S14:
	ustar = kiss(seed);
	if (ustar > tt) 
		goto S17;
S15:
	u = kiss(seed);
	if (ustar < u) {
		u = kiss(seed);
		goto S13;
	}
S16:
    tt = u;
	goto S14;
S17:
	y = aa+w;
    snorm = y;
    if(s == 1.0) snorm = -y;
    return snorm;
}

double snorm32(unsigned long *seed)
/*
 **********************************************************************
 
 
 (STANDARD-)  N O R M A L  DISTRIBUTION
 
 
 **********************************************************************
 **********************************************************************
 
 FOR DETAILS SEE:
 
 AHRENS, J.H. AND DIETER, U.
 EXTENSIONS OF FORSYTHE'S METHOD FOR RANDOM
 SAMPLING FROM THE NORMAL DISTRIBUTION.
 MATH. COMPUT., 27,124 (OCT. 1973), 927 - 937.
 
 ALL STATEMENT NUMBERS CORRESPOND TO THE STEPS OF ALGORITHM 'FL'
 (M=5) IN THE ABOVE PAPER     (SLIGHTLY MODIFIED IMPLEMENTATION)
 
 Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of
 SUNIF.  The argument IR thus goes away.
 
 **********************************************************************
 THE DEFINITIONS OF THE CONSTANTS A(K), D(K), T(K) AND
 H(K) ARE ACCORDING TO THE ABOVEMENTIONED ARTICLE
 */
{
/* FOR 32 BIT COMPILER ****************************************************/
	static double a[33] = {0,
		0.00000000000000,0.03917608550309,0.07841241273311,0.11776987457909,
		0.15731068461017,0.19709908429430,0.23720210932878,0.27769043982157,
		0.31863936396437,0.36012989178957,0.40225006532172,0.44509652498551,
		0.48877641111466,0.53340970624127,0.57913216225555,0.62609901234641,
		0.67448975019607,0.72451438349236,0.77642176114792,0.83051087820539,
		0.88714655901887,0.94678175630104,1.00999016924958,1.07751556704027,
		1.15034938037600,1.22985875921658,1.31801089730353,1.41779713799625,
		1.53412054435253,1.67593972277344,1.86273186742164,2.15387469406144
	};
	static double d[32] = {0,
		0.67448975019607,0.47585963017993,0.38377116397654,0.32861132306910,
		0.29114282663980,0.26368432217502,0.24250845238097,0.22556744380930,
		0.21163416577204,0.19992426749317,0.18991075842246,0.18122518100691,
		0.17360140038056,0.16684190866667,0.16079672918053,0.15534971747692,
		0.15040938382813,0.14590257684509,0.14177003276856,0.13796317369537,
		0.13444176150074,0.13117215026483,0.12812596512583,0.12527909006226,
		0.12261088288608,0.12010355965651,0.11774170701949,0.11551189226063,
		0.11340234879117,0.11140272044119,0.10950385201710
	};
	static double t[32] = {0,
		0.00076738283767,0.00230687039764,0.00386061844387,0.00543845406707,
		0.00705069876857,0.00870839582019,0.01042356984914,0.01220953194966,
		0.01408124734637,0.01605578804548,0.01815290075142,0.02039573175398,
		0.02281176732513,0.02543407332319,0.02830295595118,0.03146822492920,
		0.03499233438388,0.03895482964836,0.04345878381672,0.04864034918076,
		0.05468333844273,0.06184222395816,0.07047982761667,0.08113194985866,
		0.09462443534514,0.11230007889456,0.13649799954975,0.17168856004707,
		0.22762405488269,0.33049802776911,0.58470309390507
	};
	static double h[33] = {0,
		0.03920617164634,0.03932704963665,0.03950999486086,0.03975702679515,
		0.04007092772490,0.04045532602655,0.04091480886081,0.04145507115859,
		0.04208311051344,0.04280748137995,0.04363862733472,0.04458931789605,
		0.04567522779560,0.04691571371696,0.04833486978119,0.04996298427702,
		0.05183858644724,0.05401138183398,0.05654656186515,0.05953130423884,
		0.06308488965373,0.06737503494905,0.07264543556657,0.07926471414968,
		0.08781922325338,0.09930398323927,0.09930398323927,0.14043438342816,
		0.18361418337460,0.27900163464163,0.70104742502766
	};
	
	const int bits=32;
	const int log2bits=5;

	return compute_snorm(a,d,t,h,bits,log2bits,seed);
}

double snorm64(unsigned long *seed)
/*
 **********************************************************************
 
 
 (STANDARD-)  N O R M A L  DISTRIBUTION
 
 
 **********************************************************************
 **********************************************************************
 
 FOR DETAILS SEE:
 
 AHRENS, J.H. AND DIETER, U.
 EXTENSIONS OF FORSYTHE'S METHOD FOR RANDOM
 SAMPLING FROM THE NORMAL DISTRIBUTION.
 MATH. COMPUT., 27,124 (OCT. 1973), 927 - 937.
 
 ALL STATEMENT NUMBERS CORRESPOND TO THE STEPS OF ALGORITHM 'FL'
 (M=5) IN THE ABOVE PAPER     (SLIGHTLY MODIFIED IMPLEMENTATION)
 
 Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of
 SUNIF.  The argument IR thus goes away.
 
 **********************************************************************
 THE DEFINITIONS OF THE CONSTANTS A(K), D(K), T(K) AND
 H(K) ARE ACCORDING TO THE ABOVEMENTIONED ARTICLE
 */
{	
/* FOR 64 BIT COMPILER ********************************************************************************************/
	static double a[65] = {0,
		0.00000000000000000000,		0.01958428523012686884,		0.03917608550309759075,		0.05878293606894305356,
		0.07841241273311219673,		0.09807215248866102408,		0.11776987457909532386,		0.13751340214433596665,
		0.15731068461017072568,		0.17716982099173980703,		0.19709908429431238774,		0.21710694721012974151,
		0.23720210932878765808,		0.25739352610093829687,		0.27769043982157681771,		0.29810241293048683753,
		0.31863936396437531062,		0.33931160653881714540,		0.36012989178956955616,		0.38110545476355645045,
		0.40225006532172535856,		0.42357608420119957637,		0.44509652498551627309,		0.46682512285258964679,
		0.48877641111466973989,		0.51096580673824743002,		0.53340970624128070110,		0.55612559361869151608,
		0.57913216225555608219,		0.60244945316442366501,		0.62609901234642140189,		0.65010407064799524690,
		0.67448975019608170545,		0.69928330238321989576,		0.72451438349236529923,		0.75021537546794059281,
		0.77642176114792760266,		0.80317256559791805337,		0.83051087820539915008,		0.85848447414183248760,
		0.88714655901887595757,		0.91655666753311293427,		0.94678175630104566274,		0.97789754394054195785,
		1.00999016924958207042,		1.04315826331845373787,		1.07751556704028028655,		1.11319427716092844705,
		1.15034938037600831251,		1.18916435019933675044,		1.22985875921658882604,		1.27269864119053588425,
		1.31801089730353693241,		1.36620381637209842296,		1.41779713799626705395,		1.47346757794710159217,
		1.53412054435254652240,		1.60100866488607573856,		1.67593972277344405164,		1.76167041036306715185,
		1.86273186742165153262,		1.98742788592989594321,		2.15387469406145637407,		2.41755901623650526489
	};
	static double d[64] = {0,
		0.67448975019608159442,		0.47585963017992660706,		0.38377116397653820989,		0.32861132306910501022,
		0.29114282663980484145,		0.26368432217504889081,		0.24250845238095486422,		0.22556744380929671934,
		0.21163416577202820434,		0.19992426749317804280,		0.18991075842246818439,		0.18122518100689166687,
		0.17360140038058791134,		0.16684190866667414355,		0.16079672918052168029,		0.15534971747693937516,
		0.15040938382815838281,		0.14590257684504237545,		0.14177003276856758873,		0.13796317369537902664,
		0.13444176150073516851,		0.13117215026482575979,		0.12812596512584484287,		0.12527909006226956024,
		0.12261088288607169261,		0.12010355965649921473,		0.11774170701949415729,		0.11551189226063662829,
		0.11340234879117438993,		0.11140272044119647887,		0.10950385201710233218,		0.10769761656474585720,
		0.10597677198477484239,		0.10433484129316727973,		0.10276601206127988775,		0.10126505151400522209,
		0.09982723448905161945,		0.09844828202068178769,		0.09712430874765942690,		0.09585177768778052609,
		0.09462746119187670502,		0.09344840710526014504,		0.09231190933665622822,		0.09121548217292563265,
		0.09015683778984140417,		0.08913386650005250544,		0.08814461935364548140,		0.08718729276769110470,
		0.08626021491139113095,		0.08536183361501414879,		0.08449070560536142693,		0.08364548689948136939,
		0.08282492421220766232,		0.08202784725386180753,		0.08125316181108743763,		0.08049984351871941612,
		0.07976693224257047632,		0.07905352700376866437,		0.07835878138394747339,		0.07768189935859837192,
		0.07702213151212333742,		0.07637877159414330208,		0.07575115338119786657
	};
	static double t[64] = {0,
		0.00019177211398748272,		0.00057561072368552389,		0.00096033394876971650,		0.00134653644887124482,
		0.00173482031156562246,		0.00212579813230833132,		0.00252009620546701047,		0.00291835786160537028,
		0.00332124698886226153,		0.00372945177970566630,		0.00414368874862280131,		0.00456470707156203526,
		0.00499329330432413555,		0.00543027654481318539,		0.00587653411333885576,		0.00633299783632157703,
		0.00680066103216570971,		0.00728058631420701059,		0.00777391434508504717,		0.00828187370039757190,
		0.00880579202794469669,		0.00934710872348029978,		0.00990738938607655176,		0.01048834236790061741,
		0.01109183779676568664,		0.01171992952837067579,		0.01237488058266632447,		0.01305919274052685757,
		0.01377564112965868476,		0.01452731482152561614,		0.01531766470596462870,		0.01615056022323955315,
		0.01703035693620452706,		0.01796197744767021043,		0.01895100885059045406,		0.02000382079777323144,
		0.02112770947254621889,		0.02233107434418067985,		0.02362363676253765449,		0.02501671241822665467,
		0.02652355381018667649,		0.02815978463254364961,		0.02994395619022578378,		0.03189826776792762525,
		0.03404951017438651723,		0.03643031744228214519,		0.03908085074485260352,		0.04205109911381214177,
		0.04540407742677228520,		0.04922035791837829632,		0.05360463183323449510,		0.05869544706131908651,
		0.06468007122940593046,		0.07181792832035974183,		0.08047898937544599451,		0.09120957067162786813,
		0.10485145021787053987,		0.12277260466481686174,		0.14735434018942444867,		0.18314368757968788048,
		0.24004979593181113851,		0.34465329797327842742,		0.60270769963404347003
	};
	static double h[64] = {0,
		0.01958804167028678489,		0.01960308401834899225,		0.01962569778970120818,		0.01965594410934935754,
		0.01969390514220356950,		0.01973968467524929785,		0.01979340885980148540,		0.01985522712400606196,
		0.01992531326826343929,		0.02000386675905814482,		0.02009111423984027631,		0.02018731128121912036,
		0.02029274439687391018,		0.02040773335641667130,		0.02053263383206082107,		0.02066784042256602366,
		0.02081379010574565397,		0.02097096618011908345,		0.02113990276736380652,		0.02132118996056448640,
		0.02151547971926351388,		0.02172349263179985251,		0.02194602568905165002,		0.02218396124266944908,
		0.02243827735645432392,		0.02271005980351555736,		0.02300051601645413876,		0.02331099136608705269,
		0.02364298822995620358,		0.02399818842032610394,		0.02437847967934319779,		0.02478598712585501976,
		0.02522311076653368025,		0.02569257048069227556,		0.02619746027715056519,		0.02674131413585766406,
		0.02732818643336618752,		0.02796275087616453131,		0.02865042312544178321,		0.02939751403137721639,
		0.03021142281289715403,		0.03110088293321398509,		0.03207627831201681490,		0.03315005462808830183,
		0.03433726098618120753,		0.03565627306852148237,		0.03712977324710863003,		0.03878610140969741826,
		0.04066115191305032933,		0.04280109419271577176,		0.04526636901967401316,		0.04813772117806043582,
		0.05152559844619943735,		0.05558534602031824090,		0.06054286884975280947,		0.06674032257389593548,
		0.07472292788700729549,		0.08541805498303849109,		0.10054667680906168026,		0.12371999275997927481,
		0.16408445953526668015,		0.25398282713075071015,		0.66370358029129239430		
	};

	const int bits=64;
	const int log2bits=6;
	
	return compute_snorm(a,d,t,h,bits,log2bits,seed);
}

double snorm(unsigned long *seed)
{
	double x;

	switch (PREC) {
		case 32: 
			x = snorm32(seed);
			break;
		case 64: default:
			x = snorm64(seed);
			break;
	}
	return x;
}

double rnorm(double mean,double stdev,unsigned long *seed)
{
	return mean + stdev*snorm(seed);
}

double rgamma(double alpha,double beta,unsigned long *seed)
{
 return sgamma(alpha,seed)/beta;
}

double rinverse_gamma(double alpha,double beta,unsigned long *seed)
{
 return beta/sgamma(alpha,seed);
}

double fact_ln(int k)
{
  int i;
  double fact;

  if (k == 0) return 0;
  fact = 0.0;
  for (i=1;i<=k;i++)
    fact += log((double)i);
  return fact;
}

/*int rpois(double lambda,unsigned long *seed)
{
 int i;
 double U,P;

 U = kiss(seed);
 i = 0;

 P = exp(-lambda);
 while (P <= U) {
   i++;
   P += exp(-lambda + i * log(lambda) - fact_ln(i));
 }
 return i;
f}*/

int rpois(double xm, unsigned long *seed)
{
	double PI = 3.141592653589793;
	static double sq,alxm,g,oldm=(-1.0);
	double em,t,y;

	if (xm < 12.0) {
		if (xm != oldm) {
			oldm=xm;
			g=exp(-xm);
		}
		em = -1;
		t=1.0;
		do {
			++em;
			t *= kiss(seed);
		} while (t > g);
	} else {
		if (xm != oldm) {
			oldm=xm;
			sq=sqrt(2.0*xm);
			alxm=log(xm);
			g=xm*alxm-lgamma(xm+1.0);
		}
		do {
			do {
				y=tan(PI*kiss(seed));
				em=sq*y+xm;
			} while (em < 0.0);
			em=floor(em);
			t=0.9*(1.0+y*y)*exp(em*alxm-lgamma(em+1.0)-g);
		} while (kiss(seed) > t);
	}
	return (int)em;
}

double *rdirichlet(double *alpha,int len,unsigned long *seed)
{
 int i;
 double *theta,denom=0.0;

 theta = (double *)calloc(len,sizeof(double));
 for (i=0;i<len;i++) {
   theta[i] = sgamma(alpha[i],seed);
   denom += theta[i];
 }
 for (i=0;i<len;i++)
   theta[i] /= denom;

 return theta;
}

double rbeta(double alpha,double beta,unsigned long *seed)
{
 double a,b;

 a = sgamma(alpha,seed);
 b = sgamma(beta,seed);

 return a/(a+b);
}


double truncNorm2(double mu,double std,double u, double v,unsigned long *seed)
{
	/*__________________Optimal truncated normal N(mu,std), a<x<b
	 (Robert, Stat&Computing, 1995)        ____________*/
	double x,a,b,boun,boo;
	double kiss(unsigned long *);
	double truncNormLeft(double,unsigned long *);
	int stop;
	
	u = (u-mu)/std;
	v = (v-mu)/std;
	if (v<0)
	{
		a=-v; b=-u;
	}
	else
	{
		a=u; b=v;
	}
	if (b>a+3*exp(-a*a-1/(a*a))/(a+sqrt(a*a+4.0)))
	{
		stop=0;
		while (stop==0)
		{
			x=truncNormLeft(a,seed);
			if( x < b ) stop=1;
			else stop = 0;
		}
	}
	else
	{
		boo=0.0;
		if (b<0.0)
			boo=b*b;
		if (a>0.0)
			boo=a*a;
		
		
		stop=0;
		while (stop==0)
		{
			x=(b-a)*kiss(seed)+a;
			boun=boo-x*x;
			if( 2.0*log(kiss(seed))<boun ) stop=1;
			else stop=0;
		}
	}
	if (v <= 0)
		x = -x;
	return(std*x + mu);
}




double truncNormLeft(double a,unsigned long *seed)
{
	/* Computes a truncated N(0,1) on x>a */
	double x,a_star, zero=0.0, one=1.0;
	int stop;
	double rnorm(double,double,unsigned long *);
	double kiss(unsigned long *);
	
	
	if (a<0.0)
	{
		stop=0;
		while (stop==0)
		{
			x = rnorm(zero,one,seed);
			if( x>a ) stop=1;
			else stop=0;
		}
	}
	else
	{
		a_star=0.5*(a+sqrt(a*a+4.0));
		stop=0;
		while (stop==0)
		{
			x=a-log(kiss(seed))/a_star;
			if( log(kiss(seed))<x*(a_star-0.5*x)-a_star*a_star*0.5 ) stop=1;
			else stop=0;
		}
	}
	
	
	return(x);
}

inline double psi(double alpha,double lambda,double x)
{
	return -alpha*(cosh(x)-1) - lambda*(exp(x)-x-1);
}

inline double psi_prime(double alpha,double lambda,double x)
{
	return -alpha*sinh(x) - lambda*(exp(x)-1);
}

double rGIG(double lambda,double a,double b,unsigned long *seed)
/*
 **********************************************************************
 
 
 GENERALIZED INVERSE GAUSSIAN  DISTRIBUTION
 
 
 **********************************************************************
 **********************************************************************
 
 FOR DETAILS SEE:
 
 Devroye, Luc, (2012) Random variate generation for the generalized inverse Gaussian distribution, Statistics & Computing,
 early online editions
 
 **********************************************************************
 Three parameter representation
   f(x) \propto  x^{\lambda -1} \exp(-0.5*(ax +b/x))
*/
 {
	double alpha,tmp,t,s,p,r,q;
	double eta,zeta,theta,xi,U,V,W,X;
	double omega = sqrt(a*b);
	
	alpha = sqrt(omega*omega + lambda*lambda) - lambda;
	tmp = -psi(alpha,lambda,1.0);
	
	if ((0.5 <= tmp) && (tmp <= 2)) 
		t = 1;
	else if (tmp > 2)
		t = sqrt(2/(alpha + lambda));
	else 
		t = log(4/(alpha + 2*lambda));
	
	tmp = -psi(alpha,lambda,-1.0);
	
	if ((0.5 <= tmp) && (tmp <= 2)) 
		s = 1;
	else if (tmp > 2)
		s = sqrt(4/(alpha*cosh(1) + lambda));
	else {
		double a = 1/lambda;
		double b = log(1 + 1/alpha + sqrt(1/(alpha*alpha) + 2/alpha));
		s = (a < b) ?  a:b;
	}
	
	eta   = -psi(alpha,lambda,t);
	zeta  = -psi_prime(alpha,lambda,t);
	theta = -psi(alpha,lambda,-s);
	xi    =  psi_prime(alpha,lambda,-s);
	
	p = 1/xi;
	r = 1/zeta;
	t -= r*eta;
	s -= p*theta;
	q = t + s;

	do {
		U = kiss(seed);
		V = kiss(seed);
		W = kiss(seed);
		if (U < q/(p + q + r))
			X = -s + q*V;
		else if (U < (q + r)/(p + q + r))
			X = t + r*log(1/V);
		else 
			X = -s - p*log(1/V);
		if ((-s <= X) && (X <= t))
			tmp = 1;
		else if (t < X)
			tmp = exp(-eta - zeta*(X-t));
		else
			tmp = exp(-theta + xi*(X + s));
//		tmp = ((-s <= X) && (X <= t)) + (t < X)*exp(-eta - zeta*(X-t)) + (X < -s)*exp(-theta + xi*(X + s)); 
	} while (exp(psi(alpha,lambda,X)) < W*tmp);
	
	tmp = sqrt(b/a)*(lambda/omega + sqrt(1 + (lambda*lambda)/(omega*omega)))*exp(X);																				  
	if (lambda < 0)
		return 1/tmp;
	else 
		return tmp;
}

