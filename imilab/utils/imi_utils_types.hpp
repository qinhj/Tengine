// ============================================================
//                  Imilab Utils: Opencv Types
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/10
// ============================================================

#ifndef __IMI_UTILS_TYPES_HPP__
#define __IMI_UTILS_TYPES_HPP__

/* std c includes */
#include <stdint.h> // for: uint64_t
#include <assert.h> // for: assert
/* std c++ includes */
#include <algorithm> // for: std::numeric_limits

#ifdef _DEBUG
#define CV_DbgAssert assert
#else // !_DEBUG
#define CV_DbgAssert(...)
#endif // _DEBUG

#if defined _MSC_VER || defined __BORLANDC__
typedef __int64 int64;
typedef unsigned __int64 uint64;
#define CV_BIG_INT(n)  n##I64
#define CV_BIG_UINT(n) n##UI64
#else
typedef int64_t int64;
typedef uint64_t uint64;
#define CV_BIG_INT(n)  n##LL
#define CV_BIG_UINT(n) n##ULL
#endif

//template<typename _Tp> class Complex;
template<typename _Tp>
class Point_;
//template<typename _Tp> class Point3_;
template<typename _Tp>
class Size_;
template<typename _Tp>
class Rect_;
//template<typename _Tp> class Scalar_;

//////////////////////////////// Point_ ////////////////////////////////

/** @brief Template class for 2D points specified by its coordinates `x` and `y`.

An instance of the class is interchangeable with C structures, CvPoint and CvPoint2D32f . There is
also a cast operator to convert point coordinates to the specified type. The conversion from
floating-point coordinates to integer coordinates is done by rounding. Commonly, the conversion
uses this operation for each of the coordinates. Besides the class members listed in the
declaration above, the following operations on points are implemented:
@code
    pt1 = pt2 + pt3;
    pt1 = pt2 - pt3;
    pt1 = pt2 * a;
    pt1 = a * pt2;
    pt1 = pt2 / a;
    pt1 += pt2;
    pt1 -= pt2;
    pt1 *= a;
    pt1 /= a;
    double value = norm(pt); // L2 norm
    pt1 == pt2;
    pt1 != pt2;
@endcode
For your convenience, the following type aliases are defined:
@code
    typedef Point_<int> Point2i;
    typedef Point2i Point;
    typedef Point_<float> Point2f;
    typedef Point_<double> Point2d;
@endcode
Example:
@code
    Point2f a(0.3f, 0.f), b(0.f, 0.4f);
    Point pt = (a + b)*10.f;
    cout << pt.x << ", " << pt.y << endl;
@endcode
*/
template<typename _Tp>
class Point_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Point_();
    Point_(_Tp _x, _Tp _y);
    Point_(const Point_& pt);
    Point_(const Size_<_Tp>& sz);
    //Point_(const Vec<_Tp, 2>& v);

    Point_& operator=(const Point_& pt);
    //! conversion to another data type
    //template<typename _Tp2> operator Point_<_Tp2>() const;

    //! conversion to the old-style C structures
    //operator Vec<_Tp, 2>() const;

    //! dot product
    _Tp dot(const Point_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point_& pt) const;
    //! cross-product
    double cross(const Point_& pt) const;
    //! checks whether the point is inside the specified rectangle
    bool inside(const Rect_<_Tp>& r) const;
    _Tp x; //!< x coordinate of the point
    _Tp y; //!< y coordinate of the point
};

typedef Point_<int> Point2i;
typedef Point_<int64> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

//////////////////////////////// Size_ ////////////////////////////////

/** @brief Template class for specifying the size of an image or rectangle.

The class includes two members called width and height. The structure can be converted to and from
the old OpenCV structures CvSize and CvSize2D32f . The same set of arithmetic and comparison
operations as for Point_ is available.

OpenCV defines the following Size_\<\> aliases:
@code
    typedef Size_<int> Size2i;
    typedef Size2i Size;
    typedef Size_<float> Size2f;
@endcode
*/
template<typename _Tp>
class Size_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Size_();
    Size_(_Tp _width, _Tp _height);
    Size_(const Size_& sz);
    Size_(const Point_<_Tp>& pt);

    Size_& operator=(const Size_& sz);
    //! the area (width*height)
    _Tp area() const;
    //! true if empty
    bool empty() const;

    //! conversion of another data type.
    //template<typename _Tp2> operator Size_<_Tp2>() const;

    _Tp width;  //!< the width
    _Tp height; //!< the height
};

typedef Size_<int> Size2i;
typedef Size_<int64> Size2l;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

//////////////////////////////// Rect_ ////////////////////////////////

/** @brief Template class for 2D rectangles

described by the following parameters:
-   Coordinates of the top-left corner. This is a default interpretation of Rect_::x and Rect_::y
    in OpenCV. Though, in your algorithms you may count x and y from the bottom-left corner.
-   Rectangle width and height.

OpenCV typically assumes that the top and left boundary of the rectangle are inclusive, while the
right and bottom boundaries are not. For example, the method Rect_::contains returns true if

\f[x  \leq pt.x < x+width,
      y  \leq pt.y < y+height\f]

Virtually every loop over an image ROI in OpenCV (where ROI is specified by Rect_\<int\> ) is
implemented as:
@code
    for(int y = roi.y; y < roi.y + roi.height; y++)
        for(int x = roi.x; x < roi.x + roi.width; x++)
        {
            // ...
        }
@endcode
In addition to the class members, the following operations on rectangles are implemented:
-   \f$\texttt{rect} = \texttt{rect} \pm \texttt{point}\f$ (shifting a rectangle by a certain offset)
-   \f$\texttt{rect} = \texttt{rect} \pm \texttt{size}\f$ (expanding or shrinking a rectangle by a
    certain amount)
-   rect += point, rect -= point, rect += size, rect -= size (augmenting operations)
-   rect = rect1 & rect2 (rectangle intersection)
-   rect = rect1 | rect2 (minimum area rectangle containing rect1 and rect2 )
-   rect &= rect1, rect |= rect1 (and the corresponding augmenting operations)
-   rect == rect1, rect != rect1 (rectangle comparison)

This is an example how the partial ordering on rectangles can be established (rect1 \f$\subseteq\f$
rect2):
@code
    template<typename _Tp> inline bool
    operator <= (const Rect_<_Tp>& r1, const Rect_<_Tp>& r2)
    {
        return (r1 & r2) == r1;
    }
@endcode
For your convenience, the Rect_\<\> alias is available: cv::Rect
*/
template<typename _Tp>
class Rect_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
    Rect_(const Rect_& r);
    Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz);
    Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2);

    Rect_& operator=(const Rect_& r);
    //! the top-left corner
    Point_<_Tp> tl() const;
    //! the bottom-right corner
    Point_<_Tp> br() const;

    //! size (width, height) of the rectangle
    Size_<_Tp> size() const;
    //! area (width*height) of the rectangle
    _Tp area() const;
    //! true if empty
    bool empty() const;

    //! conversion to another data type
    //template<typename _Tp2> operator Rect_<_Tp2>() const;

    //! checks whether the rectangle contains the point
    bool contains(const Point_<_Tp>& pt) const;

    _Tp x;      //!< x coordinate of the top-left corner
    _Tp y;      //!< y coordinate of the top-left corner
    _Tp width;  //!< width of the rectangle
    _Tp height; //!< height of the rectangle
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;

/////////////////////////////////////////////////////////////////////////
///////////////////////////// Implementation ////////////////////////////
/////////////////////////////////////////////////////////////////////////

//////////////////////////////// 2D Point ///////////////////////////////

template<typename _Tp>
inline Point_<_Tp>::Point_()
    : x(0), y(0)
{
}

template<typename _Tp>
inline Point_<_Tp>::Point_(_Tp _x, _Tp _y)
    : x(_x), y(_y)
{
}

template<typename _Tp>
inline Point_<_Tp>::Point_(const Point_& pt)
    : x(pt.x), y(pt.y)
{
}

template<typename _Tp>
inline Point_<_Tp>::Point_(const Size_<_Tp>& sz)
    : x(sz.width), y(sz.height)
{
}
/*
template<typename _Tp> inline
Point_<_Tp>::Point_(const Vec<_Tp,2>& v)
    : x(v[0]), y(v[1]) {}
*/
template<typename _Tp>
inline Point_<_Tp>& Point_<_Tp>::operator=(const Point_& pt)
{
    x = pt.x;
    y = pt.y;
    return *this;
}
/*
template<typename _Tp> template<typename _Tp2> inline
Point_<_Tp>::operator Point_<_Tp2>() const
{
    return Point_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
}

template<typename _Tp> inline
Point_<_Tp>::operator Vec<_Tp, 2>() const
{
    return Vec<_Tp, 2>(x, y);
}

template<typename _Tp> inline
_Tp Point_<_Tp>::dot(const Point_& pt) const
{
    return saturate_cast<_Tp>(x*pt.x + y*pt.y);
}

template<typename _Tp> inline
double Point_<_Tp>::ddot(const Point_& pt) const
{
    return (double)x*(double)(pt.x) + (double)y*(double)(pt.y);
}

template<typename _Tp> inline
double Point_<_Tp>::cross(const Point_& pt) const
{
    return (double)x*pt.y - (double)y*pt.x;
}

template<typename _Tp> inline bool
Point_<_Tp>::inside( const Rect_<_Tp>& r ) const
{
    return r.contains(*this);
}


template<typename _Tp> static inline
Point_<_Tp>& operator += (Point_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator -= (Point_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator *= (Point_<_Tp>& a, int b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator *= (Point_<_Tp>& a, float b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator *= (Point_<_Tp>& a, double b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator /= (Point_<_Tp>& a, int b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator /= (Point_<_Tp>& a, float b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator /= (Point_<_Tp>& a, double b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template<typename _Tp> static inline
double norm(const Point_<_Tp>& pt)
{
    return std::sqrt((double)pt.x*pt.x + (double)pt.y*pt.y);
}
*/
template<typename _Tp>
static inline bool operator==(const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return a.x == b.x && a.y == b.y;
}

template<typename _Tp>
static inline bool operator!=(const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return a.x != b.x || a.y != b.y;
}
/*
template<typename _Tp> static inline
Point_<_Tp> operator + (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x + b.x), saturate_cast<_Tp>(a.y + b.y) );
}

template<typename _Tp> static inline
Point_<_Tp> operator - (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x - b.x), saturate_cast<_Tp>(a.y - b.y) );
}

template<typename _Tp> static inline
Point_<_Tp> operator - (const Point_<_Tp>& a)
{
    return Point_<_Tp>( saturate_cast<_Tp>(-a.x), saturate_cast<_Tp>(-a.y) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Point_<_Tp>& a, int b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (int a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(b.x*a), saturate_cast<_Tp>(b.y*a) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Point_<_Tp>& a, float b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (float a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(b.x*a), saturate_cast<_Tp>(b.y*a) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Point_<_Tp>& a, double b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (double a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(b.x*a), saturate_cast<_Tp>(b.y*a) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Matx<_Tp, 2, 2>& a, const Point_<_Tp>& b)
{
    Matx<_Tp, 2, 1> tmp = a * Vec<_Tp,2>(b.x, b.y);
    return Point_<_Tp>(tmp.val[0], tmp.val[1]);
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (const Matx<_Tp, 3, 3>& a, const Point_<_Tp>& b)
{
    Matx<_Tp, 3, 1> tmp = a * Vec<_Tp,3>(b.x, b.y, 1);
    return Point3_<_Tp>(tmp.val[0], tmp.val[1], tmp.val[2]);
}
*/
template<typename _Tp>
static inline Point_<_Tp> operator/(const Point_<_Tp>& a, int b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp>
static inline Point_<_Tp> operator/(const Point_<_Tp>& a, float b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp>
static inline Point_<_Tp> operator/(const Point_<_Tp>& a, double b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

////////////////////////////////// Size /////////////////////////////////

template<typename _Tp>
inline Size_<_Tp>::Size_()
    : width(0), height(0)
{
}

template<typename _Tp>
inline Size_<_Tp>::Size_(_Tp _width, _Tp _height)
    : width(_width), height(_height)
{
}

template<typename _Tp>
inline Size_<_Tp>::Size_(const Size_& sz)
    : width(sz.width), height(sz.height)
{
}

template<typename _Tp>
inline Size_<_Tp>::Size_(const Point_<_Tp>& pt)
    : width(pt.x), height(pt.y)
{
}

//template<typename _Tp> template<typename _Tp2> inline
//Size_<_Tp>::operator Size_<_Tp2>() const {
//    return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
//}

template<typename _Tp>
inline Size_<_Tp>& Size_<_Tp>::operator=(const Size_<_Tp>& sz)
{
    width = sz.width;
    height = sz.height;
    return *this;
}

template<typename _Tp>
inline _Tp Size_<_Tp>::area() const
{
    const _Tp result = width * height;
    CV_DbgAssert(!std::numeric_limits<_Tp>::is_integer
                 || width == 0 || result / width == height); // make sure the result fits in the return value
    return result;
}

template<typename _Tp>
inline bool Size_<_Tp>::empty() const
{
    return width <= 0 || height <= 0;
}

template<typename _Tp>
static inline Size_<_Tp>& operator*=(Size_<_Tp>& a, _Tp b)
{
    a.width *= b;
    a.height *= b;
    return a;
}

template<typename _Tp>
static inline Size_<_Tp> operator*(const Size_<_Tp>& a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp *= b;
    return tmp;
}

template<typename _Tp>
static inline Size_<_Tp>& operator/=(Size_<_Tp>& a, _Tp b)
{
    a.width /= b;
    a.height /= b;
    return a;
}

template<typename _Tp>
static inline Size_<_Tp> operator/(const Size_<_Tp>& a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp>
static inline Size_<_Tp>& operator+=(Size_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template<typename _Tp>
static inline Size_<_Tp> operator+(const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    Size_<_Tp> tmp(a);
    tmp += b;
    return tmp;
}

template<typename _Tp>
static inline Size_<_Tp>& operator-=(Size_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width -= b.width;
    a.height -= b.height;
    return a;
}

template<typename _Tp>
static inline Size_<_Tp> operator-(const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    Size_<_Tp> tmp(a);
    tmp -= b;
    return tmp;
}

template<typename _Tp>
static inline bool operator==(const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    return a.width == b.width && a.height == b.height;
}

template<typename _Tp>
static inline bool operator!=(const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    return !(a == b);
}

////////////////////////////////// Rect /////////////////////////////////

template<typename _Tp>
inline Rect_<_Tp>::Rect_()
    : x(0), y(0), width(0), height(0)
{
}

template<typename _Tp>
inline Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    : x(_x), y(_y), width(_width), height(_height)
{
}

template<typename _Tp>
inline Rect_<_Tp>::Rect_(const Rect_<_Tp>& r)
    : x(r.x), y(r.y), width(r.width), height(r.height)
{
}

template<typename _Tp>
inline Rect_<_Tp>::Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz)
    : x(org.x), y(org.y), width(sz.width), height(sz.height)
{
}

template<typename _Tp>
inline Rect_<_Tp>::Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2)
{
    x = std::min(pt1.x, pt2.x);
    y = std::min(pt1.y, pt2.y);
    width = std::max(pt1.x, pt2.x) - x;
    height = std::max(pt1.y, pt2.y) - y;
}

template<typename _Tp>
inline Rect_<_Tp>& Rect_<_Tp>::operator=(const Rect_<_Tp>& r)
{
    x = r.x;
    y = r.y;
    width = r.width;
    height = r.height;
    return *this;
}

template<typename _Tp>
inline Point_<_Tp> Rect_<_Tp>::tl() const
{
    return Point_<_Tp>(x, y);
}

template<typename _Tp>
inline Point_<_Tp> Rect_<_Tp>::br() const
{
    return Point_<_Tp>(x + width, y + height);
}

template<typename _Tp>
inline Size_<_Tp> Rect_<_Tp>::size() const
{
    return Size_<_Tp>(width, height);
}

template<typename _Tp>
inline _Tp Rect_<_Tp>::area() const
{
    const _Tp result = width * height;
    CV_DbgAssert(!std::numeric_limits<_Tp>::is_integer
                 || width == 0 || result / width == height); // make sure the result fits in the return value
    return result;
}

template<typename _Tp>
inline bool Rect_<_Tp>::empty() const
{
    return width <= 0 || height <= 0;
}

//template<typename _Tp> template<typename _Tp2> inline
//Rect_<_Tp>::operator Rect_<_Tp2>() const {
//    return Rect_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y), saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
//}

template<typename _Tp>
inline bool Rect_<_Tp>::contains(const Point_<_Tp>& pt) const
{
    return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator+=(Rect_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator-=(Rect_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator+=(Rect_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator-=(Rect_<_Tp>& a, const Size_<_Tp>& b)
{
    const _Tp width = a.width - b.width;
    const _Tp height = a.height - b.height;
    CV_DbgAssert(width >= 0 && height >= 0);
    a.width = width;
    a.height = height;
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator&=(Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp x1 = std::max(a.x, b.x);
    _Tp y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if (a.width <= 0 || a.height <= 0)
        a = Rect_<_Tp>();
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator|=(Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    if (a.empty())
    {
        a = b;
    }
    else if (!b.empty())
    {
        _Tp x1 = std::min(a.x, b.x);
        _Tp y1 = std::min(a.y, b.y);
        a.width = std::max(a.x + a.width, b.x + b.width) - x1;
        a.height = std::max(a.y + a.height, b.y + b.height) - y1;
        a.x = x1;
        a.y = y1;
    }
    return a;
}

template<typename _Tp>
static inline bool operator==(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template<typename _Tp>
static inline bool operator!=(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

template<typename _Tp>
static inline Rect_<_Tp> operator+(const Rect_<_Tp>& a, const Point_<_Tp>& b)
{
    return Rect_<_Tp>(a.x + b.x, a.y + b.y, a.width, a.height);
}

template<typename _Tp>
static inline Rect_<_Tp> operator-(const Rect_<_Tp>& a, const Point_<_Tp>& b)
{
    return Rect_<_Tp>(a.x - b.x, a.y - b.y, a.width, a.height);
}

template<typename _Tp>
static inline Rect_<_Tp> operator+(const Rect_<_Tp>& a, const Size_<_Tp>& b)
{
    return Rect_<_Tp>(a.x, a.y, a.width + b.width, a.height + b.height);
}

template<typename _Tp>
static inline Rect_<_Tp> operator-(const Rect_<_Tp>& a, const Size_<_Tp>& b)
{
    const _Tp width = a.width - b.width;
    const _Tp height = a.height - b.height;
    CV_DbgAssert(width >= 0 && height >= 0);
    return Rect_<_Tp>(a.x, a.y, width, height);
}

template<typename _Tp>
static inline Rect_<_Tp> operator&(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

template<typename _Tp>
static inline Rect_<_Tp> operator|(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c |= b;
}

/**
 * @brief measure dissimilarity between two sample sets
 *
 * computes the complement of the Jaccard Index as described in <https://en.wikipedia.org/wiki/Jaccard_index>.
 * For rectangles this reduces to computing the intersection over the union.
 */
template<typename _Tp>
static inline double jaccardDistance(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon())
    {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

#endif // !__IMI_UTILS_TYPES_HPP__
